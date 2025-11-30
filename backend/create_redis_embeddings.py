import redis
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, VectorField, TagField, GeoField
from redis.commands.search.index_definition import IndexDefinition, IndexType
import mysql.connector
from mysql.connector import Error

# --- CONFIGURATION ---
REDIS_HOST = 'localhost'
REDIS_PORT = 26379
VECTOR_DIM = 384 # Matches 'all-MiniLM-L6-v2'
BATCH_SIZE = 1000 # Process 1000 hotels at a time (adjust based on RAM)
INDEX_NAME = "hotels_idx"

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "admin",
    "password": "bKak3CVwNcHTrxuGaJPb4fgARzyxnR",
    "database": "nuitee-lite",
}

# 1. CONNECT TO REDIS
print(f"[1/5] Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
# decode_responses=False to handle binary embeddings
# Add socket timeouts for SSH tunnel connections
r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    decode_responses=False,
    socket_connect_timeout=10,
    socket_timeout=10
)
try:
    r.ping()
    print(f"[1/5] ✓ Successfully connected to Redis")
except Exception as e:
    print(f"[1/5] ⚠ Warning: Initial Redis ping failed: {e}")
    print(f"      Will attempt connection during actual operations...")

# 2. LOAD MODEL
print("[2/5] Loading sentence transformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[2/5] ✓ Model loaded successfully")

# 3. FETCH HOTELS FROM DATABASE
def fetch_hotels_count():
    """Get total count of hotels with no primary_hotel_id"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM hotel WHERE primary_hotel_id IS NULL"
        cursor.execute(query)
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Error as exc:
        raise SystemExit(f"Failed to get hotel count: {exc}") from exc

def fetch_hotels_batch(offset, limit):
    """Fetch a batch of hotels from database"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT HotelId, EPSRapidID, name, address, Country, City, latitude_double, longitude_double
        FROM hotel
        WHERE primary_hotel_id IS NULL
        LIMIT %s OFFSET %s
        """
        
        cursor.execute(query, (limit, offset))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return pd.DataFrame(rows)
    except Error as exc:
        raise SystemExit(f"Failed to fetch hotels batch: {exc}") from exc

# 4. DEFINE REDIS INDEX
def create_index():
    print(f"[4/5] Checking for existing index '{INDEX_NAME}'...")
    try:
        info = r.ft(INDEX_NAME).info()
        num_docs = info.get('num_docs', 'N/A')
        print(f"[4/5] ⚠ Index '{INDEX_NAME}' already exists.")
        print(f"      Dropping existing index to recreate with city field...")
        r.ft(INDEX_NAME).dropindex()
        print(f"      ✓ Dropped existing index")
    except:
        pass  # Index doesn't exist, which is fine
    
    # Create the index (either new or after dropping)
    print(f"[4/5] Creating new index '{INDEX_NAME}'...")
    # Note: This Redis instance has limited RediSearch support
    # Using TAG fields and VECTOR field (GEO not supported)
    # Text and location data will still be stored in the hash
    schema = [
        TagField("country"),   # Important for pre-filtering!
        TagField("city"),      # Important for candidate filtering!
        VectorField("embedding",
                    "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    })
    ]
    r.ft(INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(prefix=["hotel:"], index_type=IndexType.HASH)
    )
    print(f"[4/5] ✓ Index '{INDEX_NAME}' created successfully")

# 5. MAIN ETL PROCESS
def process_and_load():
    create_index()
    
    # Get total count
    print(f"[3/5] Connecting to MySQL database '{DB_CONFIG['database']}' at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    print(f"[3/5] Getting total count of hotels (excluding those with primary_hotel_id)...")
    total_count = fetch_hotels_count()
    print(f"[3/5] ✓ Found {total_count:,} hotels with no primary_hotel_id")
    
    total_processed = 0
    
    # Process in batches
    print(f"[5/5] Processing hotels in batches of {BATCH_SIZE}...")
    offset = 0
    
    while offset < total_count:
        # Fetch batch
        print(f"\n      Fetching batch: {offset:,} to {min(offset + BATCH_SIZE, total_count):,} of {total_count:,}...")
        df = fetch_hotels_batch(offset, BATCH_SIZE)
        
        if len(df) == 0:
            break
        
        # Handle missing values
        df['HotelId'] = df['HotelId'].fillna(0)
        df['name'] = df['name'].fillna('')
        df['address'] = df['address'].fillna('')
        df['Country'] = df['Country'].fillna('')
        df['City'] = df['City'].fillna('')
        df['latitude_double'] = df['latitude_double'].fillna(0)
        df['longitude_double'] = df['longitude_double'].fillna(0)
        
        # A. PREPARE TEXT FOR EMBEDDING
        # Combine relevant fields to give the vector semantic "texture"
        # "Hilton, 123 Main St, New York, USA" is better than just "Hilton"
        print(f"      Preparing text for embeddings (combining name, address, city, country)...")
        sentences = (
            df["name"].astype(str) + ", " + 
            df["address"].astype(str) + ", " + 
            df["City"].astype(str) + ", " +
            df["Country"].astype(str)
        ).tolist()
        
        # B. GENERATE EMBEDDINGS (Batch Operation)
        print(f"      Generating embeddings for {len(sentences)} records...")
        embeddings = model.encode(sentences).astype(np.float32)
        print(f"      ✓ Generated {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")
        
        # C. PUSH TO REDIS (Pipeline Operation)
        print(f"      Preparing Redis pipeline for {len(df)} records...")
        pipe = r.pipeline(transaction=False)
        
        for idx, row in df.iterrows():
            # Create a unique key for Redis using EPSRapidID if available, otherwise use HotelId
            if pd.notna(row['EPSRapidID']) and row['EPSRapidID']:
                key = f"hotel:{row['EPSRapidID']}"
            else:
                key = f"hotel:hotelid:{row['HotelId']}"
            
            # Redis Geo format is "lon,lat" string
            geo_str = f"{row['longitude_double']},{row['latitude_double']}"
            
            # Serialize vector to bytes
            vec_bytes = embeddings[idx].tobytes()
            
            # Add to pipeline
            pipe.hset(key, mapping={
                "hotelId": str(row['HotelId']),
                "name": str(row['name']),
                "address": str(row['address']),
                "country": str(row['Country']),
                "city": str(row['City']),
                "location": geo_str,
                "embedding": vec_bytes
            })
        
        # Execute the batch insert
        print(f"      Executing pipeline to push {len(df)} records to Redis...")
        pipe.execute()
        total_processed += len(df)
        print(f"      ✓ Batch complete. Total processed: {total_processed:,} / {total_count:,} ({100*total_processed/total_count:.1f}%)")
        
        offset += BATCH_SIZE
    
    print(f"\n[5/5] ✓ Successfully loaded {total_processed:,} hotels into Redis")

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Redis Embeddings Generation")
    print("Processing ALL hotels (excluding those with primary_hotel_id)")
    print("=" * 60)
    try:
        process_and_load()
        print("=" * 60)
        print("✓ SUCCESS: All hotels loaded with embeddings!")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"✗ ERROR: {e}")
        print("=" * 60)
        raise
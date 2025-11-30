from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List, Optional
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# This is useful for local development
load_dotenv()

app = FastAPI()

# CORS - Allow all origins for prototype
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Cloud Storage Configuration - REQUIRED for Cloud Run deployment
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", None)
print(f"🔍 Reading GCS_BUCKET_NAME from environment: {GCS_BUCKET_NAME}")
if not GCS_BUCKET_NAME:
    raise RuntimeError("GCS_BUCKET_NAME environment variable is required. Please configure a GCS bucket.")

# Initialize GCS client
try:
    gcs_client = storage.Client()
    print(f"✓ Initialized GCS client for bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize GCS client for bucket {GCS_BUCKET_NAME}: {e}")

# Job storage (in-memory, could be moved to Redis/DB for production)
jobs: Dict[str, Dict[str, Any]] = {}

# Redis Configuration - REQUIRED for Cloud Run deployment
REDIS_HOST = os.getenv("REDIS_HOST")
print(f"🔍 Reading REDIS_HOST from environment: {REDIS_HOST}")
if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST environment variable is required. Please configure Redis host.")

REDIS_PORT_STR = os.getenv("REDIS_PORT")
print(f"🔍 Reading REDIS_PORT from environment: {REDIS_PORT_STR}")
if not REDIS_PORT_STR:
    raise RuntimeError("REDIS_PORT environment variable is required. Please configure Redis port.")
try:
    REDIS_PORT = int(REDIS_PORT_STR)
except ValueError:
    raise RuntimeError(f"Invalid REDIS_PORT value: {REDIS_PORT_STR}. Must be a valid integer.")

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)  # Optional - some Redis instances don't require passwords
REDIS_PASSWORD_DISPLAY = "***" if REDIS_PASSWORD else "None"
print(f"🔍 Reading REDIS_PASSWORD from environment: {REDIS_PASSWORD_DISPLAY}")
INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "hotels_idx")
print(f"🔍 Reading REDIS_INDEX_NAME from environment: {INDEX_NAME} (default: hotels_idx)")
VECTOR_DIM = 384  # Matches 'all-MiniLM-L6-v2'
TOP_K = 5  # Return top 5 matches
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
PROCESSING_TIME_PER_HOTEL = float(os.getenv("PROCESSING_TIME_PER_HOTEL", "0.1"))
MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", "8"))

# Initialize Redis connection
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False,
    socket_connect_timeout=10,
    socket_timeout=10
)

@app.on_event("startup")
async def startup_log_config():
    """Log configuration on startup"""
    print("\n" + "="*60)
    print("🚀 Server Starting - Configuration Summary")
    print("="*60)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"🔴 Redis Host: {REDIS_HOST}")
    print(f"🔴 Redis Port: {REDIS_PORT}")
    print(f"🔴 Redis Password: {'***' if REDIS_PASSWORD else 'None (not set)'}")
    print(f"🔴 Redis Index: {INDEX_NAME}")
    print(f"⚙️  Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"⚙️  Max Worker Threads: {MAX_WORKER_THREADS}")
    print("="*60 + "\n")

@app.on_event("startup")
async def check_redis_connection():
    """Check Redis connection on startup"""
    try:
        redis_client.ping()
        print(f"✓ Redis connection successful: {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"✗ ERROR: Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        print(f"  Error: {e}")
        raise RuntimeError(f"Redis connection failed: {e}")

# Initialize embedding model (lazy load on first use)
_embedding_model = None
_embedding_model_lock = threading.Lock()  # Lock for thread-safe model access

def get_embedding_model():
    """Lazy load the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# Expected headers
EXPECTED_HEADERS = [
    'hotel_id',
    'hotel_name',
    'hotel_address',
    'country_iso_code',
    'latitude',
    'longitude'
]


def create_job(total_hotels: int = 0) -> str:
    """Create a new job and return job ID"""
    job_id = str(uuid.uuid4())
    # Don't set estimated_completion if total_hotels is 0 - will be set when CSV is parsed
    # Set to None initially, will be calculated when we know the actual count
    estimated_completion = None
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "total_hotels": total_hotels,
        "processed": 0,
        "matched_count": 0,
        "created_at": datetime.now().isoformat(),
        "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
        "processing_started_at": None,  # Track when processing actually starts
        "file_path": None,
        "error": None
    }
    return job_id


def update_job_status(job_id: str, status: str, processed: int = None, file_path: str = None, error: str = None, matched_count: int = None):
    """Update job status"""
    if job_id in jobs:
        jobs[job_id]["status"] = status
        if processed is not None:
            jobs[job_id]["processed"] = processed
        if file_path:
            jobs[job_id]["file_path"] = file_path
        if error:
            jobs[job_id]["error"] = error
        if matched_count is not None:
            jobs[job_id]["matched_count"] = matched_count
        jobs[job_id]["updated_at"] = datetime.now().isoformat()


def prepare_hotel_text(hotel: Dict[str, Any]) -> str:
    """Prepare hotel text for embedding (same format as create_redis_embeddings.py)"""
    name = str(hotel.get('hotel_name', '') or '')
    address = str(hotel.get('hotel_address', '') or '')
    country = str(hotel.get('country_iso_code', '') or '')
    # Note: CSV doesn't have city, so we'll use empty string
    city = ''
    return f"{name}, {address}, {city}, {country}"


def format_mapped_id(hotel_id) -> str:
    """Format mapped hotel ID with 'lp' prefix and convert to hex (lowercase)"""
    # Handle None, empty string, or falsy values
    if hotel_id is None or hotel_id == "":
        return ""
    
    # Convert to string and strip whitespace - handle any input type
    try:
        hotel_id_str = str(hotel_id).strip()
    except Exception:
        return ""
    
    if not hotel_id_str or hotel_id_str == "nan" or hotel_id_str.lower() == "none":
        return ""
    
    try:
        # Handle both int and float (convert float to int)
        # First try direct int conversion
        try:
            id_int = int(hotel_id_str)
        except ValueError:
            # If that fails, try float then int
            id_int = int(float(hotel_id_str))
        
        # Convert to hex (lowercase, without '0x' prefix) and add 'lp' prefix
        hex_id = format(id_int, 'x')  # 'x' gives lowercase hex
        return f"lp{hex_id}"
    except (ValueError, TypeError, OverflowError):
        # If conversion fails, return empty string
        return ""


def process_single_hotel(
    idx: int,
    hotel: Dict[str, Any],
    original_row: Dict[str, Any],
    model: SentenceTransformer
) -> tuple[int, Dict[str, Any], bool]:
    """
    Process a single hotel: generate embedding, search Redis, and build output row.
    Returns: (index, output_row_dict, was_matched)
    """
    # Prepare text for embedding
    hotel_text = prepare_hotel_text(hotel)
    
    # Generate embedding (use lock for thread safety)
    with _embedding_model_lock:
        embedding = model.encode(hotel_text).astype(np.float32)
    
    # Search for similar hotels in Redis
    country = hotel.get('country_iso_code')
    matches = search_similar_hotels(embedding, country=country, top_k=1)  # Get best match only
    
    # Start with original row data
    output_row = original_row.copy()
    
    # Add match columns (best match only)
    was_matched = False
    if matches:
        best_match = matches[0]
        raw_hotel_id = best_match.get("hotelId", "")
        formatted_id = format_mapped_id(raw_hotel_id)
        # Ensure it's stored as a string to prevent pandas from converting it
        output_row["matched_hotel_id"] = str(formatted_id) if formatted_id else ""
        output_row["matched_hotel_name"] = best_match.get("name", "")
        output_row["matched_hotel_address"] = best_match.get("address", "")
        output_row["matched_hotel_country"] = best_match.get("country", "")
        output_row["matched_hotel_city"] = best_match.get("city", "")
        output_row["matched_hotel_location"] = best_match.get("location", "")
        output_row["match_similarity"] = best_match.get("similarity", 0)
        was_matched = True
    else:
        output_row["matched_hotel_id"] = ""
        output_row["matched_hotel_name"] = ""
        output_row["matched_hotel_address"] = ""
        output_row["matched_hotel_country"] = ""
        output_row["matched_hotel_city"] = ""
        output_row["matched_hotel_location"] = ""
        output_row["match_similarity"] = 0
    
    return (idx, output_row, was_matched)


def search_similar_hotels(embedding: np.ndarray, country: Optional[str] = None, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Search Redis for similar hotels using vector similarity"""
    try:
        # Convert embedding to bytes
        query_vector = embedding.astype(np.float32).tobytes()
        
        # Build base query with optional country filter
        if country:
            # Filter by country tag
            base_query = f"@country:{{{country}}}"
        else:
            base_query = "*"
        
        # Perform vector similarity search using KNN
        # Format: (base_query)=>[KNN top_k @embedding $vector]
        query = f"({base_query})=>[KNN {top_k} @embedding $vector AS vector_score]"
        
        # Redis search - limit is already specified in KNN query
        # Access fields directly from the hash since return_fields might not work with this Redis version
        results = redis_client.ft(INDEX_NAME).search(
            query,
            query_params={"vector": query_vector}
        )
        
        matches = []
        for doc in results.docs:
            # Get vector score (distance)
            # For cosine similarity, Redis returns distance (0 = identical, 2 = opposite)
            vector_score = float(getattr(doc, 'vector_score', 2.0))
            
            # Convert distance to similarity (cosine distance -> cosine similarity)
            # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance
            similarity = 1 - vector_score
            
            if similarity >= SIMILARITY_THRESHOLD:
                # Get the Redis key (doc.id) and fetch the hash fields
                redis_key = doc.id.decode() if isinstance(doc.id, bytes) else doc.id
                hotel_data = redis_client.hgetall(redis_key)
                
                # Decode bytes to strings - handle both bytes keys and string keys
                def decode_field(key):
                    # Try bytes key first
                    value = hotel_data.get(key.encode() if isinstance(key, str) else key)
                    if value is None:
                        # Try string key
                        value = hotel_data.get(key)
                    if value is None:
                        return ""
                    if isinstance(value, bytes):
                        return value.decode('utf-8')
                    return str(value) if value else ""
                
                matches.append({
                    "hotelId": decode_field("hotelId"),
                    "name": decode_field("name"),
                    "address": decode_field("address"),
                    "country": decode_field("country"),
                    "city": decode_field("city"),
                    "location": decode_field("location"),
                    "similarity": round(similarity, 4)
                })
        
        return matches
    except Exception as e:
        # If search fails, return empty list
        print(f"Error searching Redis: {e}")
        import traceback
        traceback.print_exc()
        return []


@app.post("/api/upload")
async def upload_csv(csv: UploadFile = File(...)):
    """Upload and parse CSV file, return headers and sample data."""
    try:
        if not csv.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV file
        contents = await csv.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Get headers
        uploaded_headers = df.columns.tolist()
        
        # Get sample record
        sample_record = df.iloc[0].to_dict() if len(df) > 0 else None
        
        return {
            "message": "CSV parsed successfully",
            "headers": uploaded_headers,
            "recordCount": len(df),
            "sampleRecord": sample_record
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")


def process_hotels_background_sync(
    job_id: str,
    df: pd.DataFrame,
    header_mapping: Dict[str, str]
):
    """Synchronous background task to process hotels and generate output CSV"""
    try:
        # Update job with actual hotel count and recalculate ETA
        actual_count = len(df)
        if job_id in jobs:
            jobs[job_id]["total_hotels"] = actual_count
            # Mark processing start time first
            if not jobs[job_id].get("processing_started_at"):
                jobs[job_id]["processing_started_at"] = datetime.now().isoformat()
            # Then calculate estimated completion based on actual count
            estimated_seconds = max(1, actual_count) * PROCESSING_TIME_PER_HOTEL
            jobs[job_id]["estimated_completion"] = (datetime.now() + timedelta(seconds=estimated_seconds)).isoformat()
        
        update_job_status(job_id, "processing", processed=0)
        
        # Transform data based on header mapping
        processed_hotels = []
        for _, row in df.iterrows():
            hotel = {}
            for expected_header, mapped_header in header_mapping.items():
                hotel[expected_header] = row.get(mapped_header, None)
            processed_hotels.append(hotel)
        
        # Load embedding model
        model = get_embedding_model()
        
        # Prepare output data with original columns + match columns
        # Use a list with None placeholders to maintain order
        output_rows = [None] * len(processed_hotels)
        matched_count = 0  # Track successfully matched hotels
        processed_count = 0  # Track processed hotels count
        progress_lock = threading.Lock()  # Lock for thread-safe progress updates
        
        # Process hotels in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            # Submit all hotel processing tasks
            future_to_idx = {
                executor.submit(
                    process_single_hotel,
                    idx,
                    hotel,
                    df.iloc[idx].to_dict(),
                    model
                ): idx
                for idx, hotel in enumerate(processed_hotels)
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_idx):
                try:
                    idx, output_row, was_matched = future.result()
                    output_rows[idx] = output_row  # Store in correct position to maintain order
                    
                    # Update progress thread-safely
                    with progress_lock:
                        processed_count += 1
                        if was_matched:
                            matched_count += 1
                        
                        # Update job status periodically (every 10 hotels or on last hotel)
                        if processed_count % 10 == 0 or processed_count == len(processed_hotels):
                            update_job_status(job_id, "processing", processed=processed_count)
                            
                            # Recalculate ETA based on actual processing speed
                            if job_id in jobs and jobs[job_id].get("processing_started_at"):
                                elapsed_time = (datetime.now() - datetime.fromisoformat(jobs[job_id]["processing_started_at"])).total_seconds()
                                
                                # Only recalculate if we have meaningful data
                                if processed_count > 0 and elapsed_time >= 1.0:
                                    hotels_per_second = processed_count / elapsed_time
                                    remaining_hotels = len(processed_hotels) - processed_count
                                    
                                    if hotels_per_second > 0 and remaining_hotels > 0:
                                        estimated_seconds_remaining = remaining_hotels / hotels_per_second
                                        if estimated_seconds_remaining > 0:
                                            new_estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                                            jobs[job_id]["estimated_completion"] = new_estimated_completion.isoformat()
                except Exception as e:
                    # Log error but continue processing other hotels
                    idx = future_to_idx[future]
                    print(f"Error processing hotel at index {idx}: {e}")
                    # Create a default output row for failed hotel
                    output_rows[idx] = df.iloc[idx].to_dict()
                    for col in ["matched_hotel_id", "matched_hotel_name", "matched_hotel_address", 
                               "matched_hotel_country", "matched_hotel_city", "matched_hotel_location", "match_similarity"]:
                        output_rows[idx][col] = "" if col != "match_similarity" else 0
                    
                    # Still count it as processed
                    with progress_lock:
                        processed_count += 1
                        if processed_count % 10 == 0 or processed_count == len(processed_hotels):
                            update_job_status(job_id, "processing", processed=processed_count)
        
        # Final progress update
        update_job_status(job_id, "processing", processed=len(processed_hotels))
        
        # Create output DataFrame
        output_df = pd.DataFrame(output_rows)
        
        # Ensure matched_hotel_id is explicitly a string type to prevent pandas from converting it
        if "matched_hotel_id" in output_df.columns:
            # Convert to string and ensure formatting is applied
            def ensure_formatted(val):
                if pd.isna(val) or val == "" or val is None:
                    return ""
                val_str = str(val).strip()
                # If it doesn't start with 'lp', it might not be formatted - try to format it
                if val_str and not val_str.startswith("lp"):
                    try:
                        # Try to convert and format
                        if '.' in val_str:
                            id_int = int(float(val_str))
                        else:
                            id_int = int(val_str)
                        hex_id = format(id_int, 'x')
                        return f"lp{hex_id}"
                    except:
                        return val_str
                return val_str
            
            output_df["matched_hotel_id"] = output_df["matched_hotel_id"].apply(ensure_formatted).astype(str)
        
        # Save to CSV file in GCS
        filename = f"{job_id}.csv"
        
        try:
            bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(filename)
            
            # Convert DataFrame to CSV string
            csv_string = output_df.to_csv(index=False)
            # Upload with public read access
            blob.upload_from_string(
                csv_string, 
                content_type='text/csv',
                predefined_acl='publicRead'
            )
            
            file_path = f"gs://{GCS_BUCKET_NAME}/{filename}"
            print(f"Job {job_id} - Uploaded file to GCS: {file_path}")
        except Exception as e:
            error_msg = f"Failed to upload file to GCS: {e}"
            print(f"Error uploading to GCS: {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
        
        # Calculate processing time metrics
        processing_end_time = datetime.now()
        if job_id in jobs:
            # Try to get processing_started_at, fallback to created_at if not available
            processing_start_time_str = jobs[job_id].get("processing_started_at")
            if not processing_start_time_str:
                processing_start_time_str = jobs[job_id].get("created_at")
                print(f"Job {job_id} - Using created_at as fallback for processing time calculation")
            
            if processing_start_time_str:
                processing_start_time = datetime.fromisoformat(processing_start_time_str)
                total_processing_time_seconds = (processing_end_time - processing_start_time).total_seconds()
                time_per_hotel_seconds = total_processing_time_seconds / len(processed_hotels) if len(processed_hotels) > 0 else 0
                
                # Store processing time metrics
                jobs[job_id]["processing_end_time"] = processing_end_time.isoformat()
                jobs[job_id]["total_processing_time_seconds"] = round(total_processing_time_seconds, 2)
                jobs[job_id]["time_per_hotel_seconds"] = round(time_per_hotel_seconds, 4)
                
                print(f"Job {job_id} - Processing time: {total_processing_time_seconds:.2f}s, Time per hotel: {time_per_hotel_seconds:.4f}s")
            else:
                print(f"Warning: Job {job_id} - Neither processing_started_at nor created_at found, cannot calculate processing time")
        
        # Update job status to completed with coverage information
        update_job_status(job_id, "completed", processed=len(processed_hotels), file_path=file_path, matched_count=matched_count)
        
        # Verify processing time is still in job dict after update
        if job_id in jobs:
            total_time = jobs[job_id].get("total_processing_time_seconds")
            if total_time is not None:
                print(f"Job {job_id} - ✓ Verified processing time in job dict after update: {total_time}s")
            else:
                print(f"Job {job_id} - ✗ WARNING: Processing time NOT in job dict after update!")
                print(f"  Job keys after update: {list(jobs[job_id].keys())}")
                # Try to recalculate as fallback
                processing_start_time_str = jobs[job_id].get("processing_started_at") or jobs[job_id].get("created_at")
                if processing_start_time_str:
                    processing_start_time = datetime.fromisoformat(processing_start_time_str)
                    total_processing_time_seconds = (datetime.now() - processing_start_time).total_seconds()
                    time_per_hotel_seconds = total_processing_time_seconds / len(processed_hotels) if len(processed_hotels) > 0 else 0
                    jobs[job_id]["total_processing_time_seconds"] = round(total_processing_time_seconds, 2)
                    jobs[job_id]["time_per_hotel_seconds"] = round(time_per_hotel_seconds, 4)
                    print(f"  Recalculated and stored: {total_processing_time_seconds:.2f}s")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing job {job_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        update_job_status(job_id, "failed", error=error_msg)


@app.post("/api/process")
async def process_hotels(
    background_tasks: BackgroundTasks,
    csv: UploadFile = File(...),
    headerMapping: str = Form(...)
):
    """Create a job to process CSV with header mapping and match hotels using Redis vector search."""
    try:
        # Parse header mapping
        header_mapping = json.loads(headerMapping)
        
        if not header_mapping or not csv:
            raise HTTPException(
                status_code=400,
                detail="Missing headerMapping or CSV file"
            )
        
        # Create job IMMEDIATELY (before reading file)
        job_id = create_job(0)  # Will be updated in background task
        
        # Start background processing - file will be read in background task
        # FastAPI keeps the file open until request completes, so we can read it in background
        async def process_with_file():
            # Read file in background (this is async but won't block the response)
            contents = await csv.read()
            # Parse CSV in thread pool to avoid blocking event loop
            df = await asyncio.to_thread(pd.read_csv, pd.io.common.BytesIO(contents))
            # Run the entire processing in a thread pool so it doesn't block the event loop
            await asyncio.to_thread(process_hotels_background_sync, job_id, df, header_mapping)
        
        background_tasks.add_task(process_with_file)
        
        # Return immediately without waiting for file read
        job_info = jobs[job_id]
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "total_hotels": job_info["total_hotels"],
            "estimated_completion": job_info["estimated_completion"],
            "message": "Job created successfully"
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid headerMapping JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "total_hotels": job["total_hotels"],
        "processed": job["processed"],
        "progress_percentage": round((job["processed"] / job["total_hotels"] * 100) if job["total_hotels"] > 0 else 0, 2),
        "created_at": job["created_at"],
        "estimated_completion": job["estimated_completion"]
    }
    
    if job["status"] == "completed":
        response["download_url"] = f"/api/download/{job_id}"
        response["file_path"] = job["file_path"]
        # Add coverage information
        matched_count = job.get("matched_count", 0)
        total_hotels = job.get("total_hotels", 0)
        response["matched_count"] = matched_count
        response["coverage_percentage"] = round((matched_count / total_hotels * 100) if total_hotels > 0 else 0, 2)
        # Add processing time information
        total_processing_time = job.get("total_processing_time_seconds")
        time_per_hotel = job.get("time_per_hotel_seconds")
        
        print(f"API Response for job {job_id} - Checking processing time:")
        print(f"  - total_processing_time_seconds: {total_processing_time}")
        print(f"  - time_per_hotel_seconds: {time_per_hotel}")
        print(f"  - Job keys: {list(job.keys())}")
        
        # If processing time is missing, try to calculate it now as fallback
        if total_processing_time is None:
            print(f"  ⚠ Processing time missing, attempting to calculate...")
            processing_start_time_str = job.get("processing_started_at") or job.get("created_at")
            if processing_start_time_str:
                try:
                    processing_start_time = datetime.fromisoformat(processing_start_time_str)
                    processing_end_time = datetime.now()
                    total_processing_time_seconds = (processing_end_time - processing_start_time).total_seconds()
                    total_hotels = job.get("total_hotels", 0)
                    time_per_hotel_seconds = total_processing_time_seconds / total_hotels if total_hotels > 0 else 0
                    
                    # Store it for future requests
                    jobs[job_id]["total_processing_time_seconds"] = round(total_processing_time_seconds, 2)
                    jobs[job_id]["time_per_hotel_seconds"] = round(time_per_hotel_seconds, 4)
                    jobs[job_id]["processing_end_time"] = processing_end_time.isoformat()
                    
                    total_processing_time = round(total_processing_time_seconds, 2)
                    time_per_hotel = round(time_per_hotel_seconds, 4)
                    print(f"  ✓ Calculated processing time: {total_processing_time}s")
                except Exception as e:
                    print(f"  ✗ Error calculating processing time: {e}")
        
        if total_processing_time is not None:
            response["total_processing_time_seconds"] = total_processing_time
            response["time_per_hotel_seconds"] = time_per_hotel
            response["processing_started_at"] = job.get("processing_started_at")
            response["processing_end_time"] = job.get("processing_end_time")
            print(f"  ✓ Including processing time in response: {total_processing_time}s")
        else:
            print(f"  ✗ NOT including processing time - could not calculate")
    
    if job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")
    
    return response


@app.get("/api/download/{job_id}")
async def download_file(job_id: str):
    """Download the processed CSV file"""
    print(f"📥 Download request for job_id: {job_id}")
    
    if job_id not in jobs:
        print(f"  ✗ Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        print(f"  ✗ Job not completed. Status: {job['status']}")
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job['status']}"
        )
    
    file_path = job.get("file_path")
    if not file_path:
        print(f"  ✗ File path not found in job")
        raise HTTPException(status_code=404, detail="Output file not found")
    
    print(f"  📁 File path: {file_path}")
    
    # All files should be in GCS
    if not file_path.startswith("gs://"):
        error_msg = f"Invalid file path format (expected gs://): {file_path}"
        print(f"  ✗ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    print(f"  ☁️  Serving from Google Cloud Storage")
    
    try:
        # Extract bucket and blob name from gs:// path
        path_parts = file_path.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else f"{job_id}.csv"
        
        print(f"  📦 Bucket: {bucket_name}, Blob: {blob_name}")
        
        # Use public URL for public bucket (no signed URL needed)
        # URL format: https://storage.googleapis.com/{bucket}/{blob}
        public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        
        print(f"  ✓ Using public URL for GCS download")
        print(f"  🔗 Redirecting to: {public_url}")
        
        return RedirectResponse(url=public_url)
    except Exception as e:
        print(f"  ✗ Error generating GCS download URL: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating download URL: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Hotel Mapping Backend API"}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run(app, host=host, port=port)


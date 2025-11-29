from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
from typing import Dict, Any

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected headers
EXPECTED_HEADERS = [
    'hotel_id',
    'hotel_name',
    'hotel_address',
    'country_iso_code',
    'latitude',
    'longitude'
]


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


@app.post("/api/process")
async def process_hotels(
    csv: UploadFile = File(...),
    headerMapping: str = Form(...)
):
    """Process CSV with header mapping and return transformed hotel data."""
    try:
        # Parse header mapping
        header_mapping = json.loads(headerMapping)
        
        if not header_mapping or not csv:
            raise HTTPException(
                status_code=400,
                detail="Missing headerMapping or CSV file"
            )
        
        # Read CSV file
        contents = await csv.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Transform data based on header mapping
        processed_hotels = []
        for _, row in df.iterrows():
            hotel = {}
            for expected_header, mapped_header in header_mapping.items():
                hotel[expected_header] = row.get(mapped_header, None)
            processed_hotels.append(hotel)
        
        # For now, just return the processed data
        # Mapping logic with embeddings/transformers will be added later
        return {
            "message": "Data processed successfully",
            "hotels": processed_hotels,
            "count": len(processed_hotels)
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid headerMapping JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing hotel data: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Hotel Mapping Backend API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)


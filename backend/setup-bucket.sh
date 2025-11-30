#!/bin/bash

# Script to create and configure Google Cloud Storage bucket
set -e

PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="${PROJECT_ID}-hotel-mapping-outputs"
REGION="us-central1"

echo "Setting up Google Cloud Storage bucket..."
echo "Project: ${PROJECT_ID}"
echo "Bucket: ${BUCKET_NAME}"
echo "Region: ${REGION}"

# Check if bucket exists
if gsutil ls -b gs://${BUCKET_NAME} 2>/dev/null; then
    echo "Bucket ${BUCKET_NAME} already exists"
else
    echo "Creating bucket ${BUCKET_NAME}..."
    gsutil mb -p ${PROJECT_ID} -c STANDARD -l ${REGION} gs://${BUCKET_NAME}
fi

# Set bucket permissions (make files publicly readable via signed URLs)
echo "Configuring bucket permissions..."
gsutil iam ch allUsers:objectViewer gs://${BUCKET_NAME} || echo "Note: Bucket will use signed URLs for access"

# Set CORS if needed (for direct browser access)
echo "Configuring CORS..."
cat > /tmp/cors.json <<EOF
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Content-Disposition"],
    "maxAgeSeconds": 3600
  }
]
EOF
gsutil cors set /tmp/cors.json gs://${BUCKET_NAME}
rm /tmp/cors.json

# Set lifecycle policy (optional: delete files older than 30 days)
echo "Setting lifecycle policy..."
cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF
gsutil lifecycle set /tmp/lifecycle.json gs://${BUCKET_NAME}
rm /tmp/lifecycle.json

echo "Bucket setup complete!"
echo "Bucket name: ${BUCKET_NAME}"
echo "Use this bucket name in your deployment: GCS_BUCKET_NAME=${BUCKET_NAME}"


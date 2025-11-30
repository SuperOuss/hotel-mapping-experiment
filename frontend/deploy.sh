#!/bin/bash

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'latest')"
SERVICE_NAME="hotel-mapping-frontend"
GCP_PROJECT="nuitee-lite-api"
REGION="us-east1"  # Will be detected if service exists
ARTIFACT_REGISTRY_REPO="docker-repo"  # Artifact Registry repository name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting Frontend Cloud Run Deployment ===${NC}"
echo -e "Service: ${SERVICE_NAME}"
echo -e "Commit: ${COMMIT}"
echo -e "Project: ${GCP_PROJECT}"
echo ""

# Set GCP project
echo -e "${GREEN}Setting GCP project to ${GCP_PROJECT}...${NC}"
gcloud config set project "${GCP_PROJECT}"

# Try to detect region from existing service
echo -e "${GREEN}Detecting service region...${NC}"
for r in us-east1 us-central1 us-west1 europe-west1 asia-northeast1; do
    if gcloud run services describe "${SERVICE_NAME}" --region="${r}" --format='value(metadata.name)' >/dev/null 2>&1; then
        REGION="${r}"
        echo -e "${GREEN}Found service in region: ${REGION}${NC}"
        break
    fi
done

# Set image name using detected/configured region
IMAGE_NAME="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}"
echo -e "${GREEN}Using image: ${IMAGE_NAME}${NC}"

# Get backend URL - check environment variable first, then try to detect from existing backend service
BACKEND_SERVICE="hotel-mapping-backend"
BACKEND_URL=""

# Check if BACKEND_URL is set as environment variable
if [ -n "${BACKEND_URL_ENV:-}" ]; then
    BACKEND_URL="${BACKEND_URL_ENV}"
    echo -e "${GREEN}Using backend URL from BACKEND_URL_ENV: ${BACKEND_URL}${NC}"
else
    # Try to detect from existing backend service
    echo -e "${GREEN}Detecting backend URL...${NC}"
    for r in us-east1 us-central1 us-west1 europe-west1 asia-northeast1; do
        BACKEND_URL=$(gcloud run services describe "${BACKEND_SERVICE}" --region="${r}" --format='value(status.url)' 2>/dev/null || echo "")
        if [ -n "$BACKEND_URL" ]; then
            echo -e "${GREEN}Found backend URL: ${BACKEND_URL}${NC}"
            break
        fi
    done
    
    # If backend URL not found, use default
    if [ -z "$BACKEND_URL" ]; then
        echo -e "${YELLOW}Backend service not found. Using default URL.${NC}"
        BACKEND_URL="https://hotel-mapping-backend-844355989729.us-east1.run.app"
        echo -e "${YELLOW}Using backend URL: ${BACKEND_URL}${NC}"
        echo -e "${YELLOW}You can override this by setting BACKEND_URL_ENV environment variable${NC}"
    fi
fi

echo ""

# Enable required APIs
echo -e "${GREEN}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com --quiet || true

# Create Artifact Registry repository if it doesn't exist
echo -e "${GREEN}Ensuring Artifact Registry repository exists...${NC}"
if ! gcloud artifacts repositories describe "${ARTIFACT_REGISTRY_REPO}" \
    --location="${REGION}" \
    --repository-format=docker 2>/dev/null; then
    echo -e "${GREEN}Creating Artifact Registry repository...${NC}"
    gcloud artifacts repositories create "${ARTIFACT_REGISTRY_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="Docker repository for ${SERVICE_NAME}" \
        --quiet || true
fi

# Configure Docker authentication for Artifact Registry
echo -e "${GREEN}Configuring Docker authentication...${NC}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Checkout the specific commit if we're building
SKIP_BUILD=false
if [[ "$1" == "--skip-build" ]] || [[ "$1" == "-s" ]]; then
    SKIP_BUILD=true
fi

# Build and push Docker image using Cloud Build (unless skipped)
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${GREEN}Building and pushing Docker image to Artifact Registry...${NC}"
    echo -e "${GREEN}Using backend URL: ${BACKEND_URL}${NC}"
    
    # Use cloudbuild.yaml with substitutions
    gcloud builds submit \
        --config=cloudbuild.yaml \
        --substitutions=_VITE_API_BASE_URL="${BACKEND_URL}",_IMAGE_NAME="${IMAGE_NAME}",COMMIT_SHA="${COMMIT}" \
        .
else
    echo -e "${YELLOW}Skipping build, using existing image: ${IMAGE_NAME}:latest${NC}"
fi

# Deploy to Cloud Run
echo -e "${GREEN}Deploying to Cloud Run...${NC}"

# Build deploy command
DEPLOY_CMD="gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 80 \
    --memory 512Mi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0"

# Execute deploy command
eval $DEPLOY_CMD

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
FRONTEND_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo -e "Frontend URL: ${FRONTEND_URL}"
echo -e "Backend URL: ${BACKEND_URL}"


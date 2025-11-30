#!/bin/bash

set -e

# Configuration
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'latest')"
SERVICE_NAME="hotel-mapping-backend"
GCP_PROJECT="nuitee-lite-api"
REGION="us-east1"  # Will be detected if service exists
ARTIFACT_REGISTRY_REPO="docker-repo"  # Artifact Registry repository name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Starting Cloud Run Deployment ===${NC}"
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

# Detect or use VPC connector for lite-api-vpc
echo -e "${GREEN}Checking for VPC connector...${NC}"
VPC_CONNECTOR=$(gcloud compute networks vpc-access connectors list \
    --region="${REGION}" \
    --format="value(name)" \
    --filter="network:lite-api-vpc AND state:READY" \
    --project="${GCP_PROJECT}" 2>/dev/null | head -1)

if [ -z "$VPC_CONNECTOR" ]; then
    # Fallback to any ready connector in the region
    VPC_CONNECTOR=$(gcloud compute networks vpc-access connectors list \
        --region="${REGION}" \
        --format="value(name)" \
        --filter="state:READY" \
        --project="${GCP_PROJECT}" 2>/dev/null | head -1)
fi

if [ -z "$VPC_CONNECTOR" ]; then
    echo -e "${YELLOW}Warning: No ready VPC connector found. Deploying without VPC access.${NC}"
else
    # Use full path format for Cloud Run: projects/PROJECT_ID/locations/REGION/connectors/CONNECTOR_NAME
    VPC_CONNECTOR_FULL="projects/${GCP_PROJECT}/locations/${REGION}/connectors/${VPC_CONNECTOR}"
    echo -e "${GREEN}Found VPC connector: ${VPC_CONNECTOR}${NC}"
    echo -e "${GREEN}Using VPC connector path: ${VPC_CONNECTOR_FULL}${NC}"
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

# Check if --skip-build flag is passed
SKIP_BUILD=false
if [[ "$1" == "--skip-build" ]] || [[ "$1" == "-s" ]]; then
    SKIP_BUILD=true
fi

# Build and push Docker image using Cloud Build (unless skipped)
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${GREEN}Building and pushing Docker image to Artifact Registry...${NC}"
    gcloud builds submit --tag "${IMAGE_NAME}:${COMMIT}" --tag "${IMAGE_NAME}:latest" .
else
    echo -e "${YELLOW}Skipping build, using existing image: ${IMAGE_NAME}:latest${NC}"
fi

# Set environment variables
BUCKET_NAME="${GCP_PROJECT}-hotel-mapping-outputs"
ENV_VARS="REDIS_HOST=10.103.221.171,REDIS_PORT=6379,GCS_BUCKET_NAME=${BUCKET_NAME},SIMILARITY_THRESHOLD=0.7,MAX_WORKER_THREADS=8"

# Deploy to Cloud Run
echo -e "${GREEN}Deploying to Cloud Run...${NC}"
echo -e "${YELLOW}Setting environment variables...${NC}"

# Build deploy command
DEPLOY_CMD="gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --set-env-vars \"${ENV_VARS}\" \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --min-instances 0"

# Add VPC connector if available
if [ -n "$VPC_CONNECTOR" ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --vpc-connector ${VPC_CONNECTOR_FULL} --vpc-egress private-ranges-only"
    echo -e "${GREEN}Using VPC connector: ${VPC_CONNECTOR_FULL} (private traffic routed through VPC)${NC}"
fi

# Execute deploy command
eval $DEPLOY_CMD

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo -e "Service URL: $(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')"

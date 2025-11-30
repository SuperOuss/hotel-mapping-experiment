FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p output

# Expose port (Cloud Run sets PORT env var)
EXPOSE 8080

# Run the application (Cloud Run provides PORT env var)
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}


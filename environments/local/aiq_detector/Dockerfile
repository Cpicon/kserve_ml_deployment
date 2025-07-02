FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model code
COPY model.py .
# COPY Procfile .
# COPY .python-version .

# Set environment variables
ENV PORT=8080

# Run the model server
CMD ["python", "model.py", "--model_name=aiq-detector", "--enable_docs_url=true"] 
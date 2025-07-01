# Stage 1: Build environment
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Create package directories and copy dependency files
RUN mkdir -p aiq_circular_detection config
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
RUN uv sync --locked --no-dev

# Copy application code
COPY config/ ./config/
COPY aiq_circular_detection/ ./aiq_circular_detection/

# Stage 2: Runtime environment
FROM python:3.12-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /data/images /logs && \
    chown -R appuser:appuser /app /data /logs

# Set working directory
WORKDIR /app

# Copy uv from builder
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin/:$PATH"

# Copy application and virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app /app

# Set Python path
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set environment defaults
ENV LOG_JSON=true \
    STORAGE_ROOT=/data/images \
    LOG_FILE=/logs/app.log

# Health check - uses curl which is installed in the runtime image
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "fastapi", "run", "aiq_circular_detection", "--port", "8000"]
# Multi-stage Docker build for production-grade Gap Trading System
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    DATA_PATH="/app/data" \
    LOG_LEVEL="INFO"

# Create non-root user
RUN groupadd -r gaptrader && useradd -r -g gaptrader gaptrader

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY gappers/ ./gappers/
COPY app.py ./
COPY README.md ./

# Create data directory and set permissions
RUN mkdir -p /app/data /app/logs \
    && chown -R gaptrader:gaptrader /app

# Switch to non-root user
USER gaptrader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import gappers; print('OK')" || exit 1

# Expose ports
EXPOSE 8501

# Default command (Streamlit app)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development stage
FROM builder as development

# Install development dependencies
RUN poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Switch to non-root user
USER gaptrader

# Development command
CMD ["python", "-m", "gappers.cli", "--help"]
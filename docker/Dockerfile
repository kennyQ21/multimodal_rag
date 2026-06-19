FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    gcc \
    g++ \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory (to be mounted)
RUN mkdir -p /models

# Copy the rest of the application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

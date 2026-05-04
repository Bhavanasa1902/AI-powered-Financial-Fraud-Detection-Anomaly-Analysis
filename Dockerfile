# Use official Python 3.10 slim as base image for a smaller footprint
FROM python:3.10-slim

# Set environment variables for production
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
# libgomp1 is required for XGBoost/LightGBM multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size small by not caching wheels locally
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the ports for the E-Commerce front-end and the Bank Dashboard
EXPOSE 5001 5002

# We default to running the shop application
CMD ["python", "src/api/app.py"]

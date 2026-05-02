# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (CPU-only to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container
COPY . .

# Set PYTHONPATH so the app can find the 'rppg' and 'api' modules
ENV PYTHONPATH=/app

# Start the app using the 'api' folder path
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
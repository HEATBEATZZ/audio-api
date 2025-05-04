FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make directories for file storage
RUN mkdir -p uploads processed

# Expose port
EXPOSE $PORT

# Start the application with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app

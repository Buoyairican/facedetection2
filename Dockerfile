# Use a small official Python image
FROM python:3.10-slim

# Install system dependencies needed by OpenCV and Torch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Set the command to run the Flask app
CMD ["python", "api.py"]

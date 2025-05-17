FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/ckpts /app/faceutils/dlibutils

# Copy model and dlib files that exist
COPY ckpts/G.pth /app/ckpts/
COPY faceutils/dlibutils/shape_predictor_68_face_landmarks.dat /app/faceutils/dlibutils/

# Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

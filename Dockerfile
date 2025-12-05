# Use a lightweight Python image
FROM python:3.9-slim-bullseye 

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and necessary files
COPY src/ src/
COPY params.yaml .

# Command to run the training script
CMD ["python", "src/train.py"]
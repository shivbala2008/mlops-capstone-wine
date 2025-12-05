# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and necessary files
COPY src/ src/
COPY params.yaml .
# In a real scenario, you pull the model from DVC remote here. 
# For this capstone, we will run the training inside the container 
# or copy the local script to run it.

# Command to run the training script (simulating an inference execution or retraining)
CMD ["python", "src/train.py"]

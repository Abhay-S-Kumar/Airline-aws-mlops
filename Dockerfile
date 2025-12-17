# Start with a lightweight Linux + Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /opt/ml/code

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------- CHANGES START HERE ----------------
# Copy model.py instead of train.py
COPY model.py .

# Define the command to run model.py when the container starts
ENTRYPOINT ["python", "model.py"]
# ---------------- CHANGES END HERE ----------------
    
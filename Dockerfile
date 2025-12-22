# Start with a lightweight Linux + Python base image
FROM python:3.9-slim

# This line forces a cache update - Date: Dec 22
ENV REFRESH_DATE=2025-12-22

RUN pip install --no-cache-dir pandas scikit-learn joblib
# ... (rest of your file stays the same)
# ---------------- CHANGES START HERE ----------------
# Copy model.py instead of train.py
COPY model.py .

# Define the command to run model.py when the container starts
ENTRYPOINT ["python", "model.py"]
# ---------------- CHANGES END HERE ----------------
    
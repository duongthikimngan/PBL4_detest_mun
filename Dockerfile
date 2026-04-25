FROM python:3.10-slim

# System deps for OpenCV + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer cache)
COPY web/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source + model source (code only, no model files)
COPY web/ .
COPY acne-lds-main/ acne-lds-main/

# HF Spaces port
EXPOSE 7860

CMD ["python", "app.py"]


FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY run.sh solution.py /app/
COPY models/graybox.joblib /app/models/graybox.joblib
RUN chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]

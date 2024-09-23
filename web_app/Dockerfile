
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir -p /app/logs /app/model_registry /app/.cache/clip /app/final_aug_dataset && chmod -R 777 /app/logs 

ENV MODEL_REGISTRY_DIR=/app/model_registry
ENV FINAL_AUG_DATASET=/app/final_aug_dataset
ENV LOG_DIR=/app/logs
ENV CLIP_CACHE_DIR=/app/.cache/clip
EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]


FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir -p /app/registry /app/model_registry /app/final_aug_dataset /app/.cache/clip

ENV REGISTRY_DIR=/app/registry
ENV MODEL_REGISTRY_DIR=/app/model_registry
ENV FINAL_AUG_DATASET=/app/final_aug_dataset
ENV CLIP_CACHE_DIR=/app/.cache/clip

CMD ["python3", "main.py"]

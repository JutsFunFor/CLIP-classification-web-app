# Base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# # Copy the rest of your training pipeline code into the container
# COPY ./main.py ./model.py ./test.py ./train.py ./utils.py ./dataset.py ./config_docker.yaml /app/

# Ensure that the container has these directories (they will be overridden by mounted volumes if they exist on the host)
RUN mkdir -p /app/registry /app/model_registry /app/final_aug_dataset /app/.cache/clip

# Set environment variables for the registry and model registry paths
ENV REGISTRY_DIR=/app/registry
ENV MODEL_REGISTRY_DIR=/app/model_registry
ENV FINAL_AUG_DATASET=/app/final_aug_dataset
ENV CLIP_CACHE_DIR=/app/.cache/clip

EXPOSE 6006

# Command to start your training pipeline
CMD ["python3", "main.py"]

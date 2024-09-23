# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application code into the container
# COPY ./app.py /app/

# Ensure that the container has the required directories
RUN mkdir -p /app/logs /app/model_registry /app/.cache/clip /app/final_aug_dataset && chmod -R 777 /app/logs 

# Set environment variables for paths
ENV MODEL_REGISTRY_DIR=/app/model_registry
ENV FINAL_AUG_DATASET=/app/final_aug_dataset
ENV LOG_DIR=/app/logs
ENV CLIP_CACHE_DIR=/app/.cache/clip
# Expose the necessary port for the Streamlit app
EXPOSE 8502

# Command to start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

import os
import streamlit as st
import torch
import clip
import pandas as pd
from PIL import Image
from torchvision import transforms
import time
import psutil
from torch.cuda import memory_allocated
from fuzzywuzzy import fuzz
import torch.nn as nn
import logging
import json_log_formatter

# Set up logging
log_dir = os.getenv("LOG_DIR", "./logs")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'inference.log')

# JSON log formatter setup
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.FileHandler(log_file_path)
json_handler.setFormatter(formatter)

logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)
logger.addHandler(json_handler)
logger.propagate = False

class AttentionHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(AttentionHead, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) 
        attn_weights = torch.softmax(self.attention(x), dim=1)
        attended_features = torch.sum(attn_weights * x, dim=1)
        out = self.classifier(attended_features) 
        return out

class CLIPFineTunerWithAttention(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTunerWithAttention, self).__init__()
        self.model = model
        self.attention_head = AttentionHead(model.visual.output_dim, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.attention_head(features)


def load_clip_model(device):
    clip_cache_dir = os.getenv("CLIP_CACHE_DIR", "/app/.cache/clip")
    os.makedirs(clip_cache_dir, exist_ok=True)
    
    # Download the model to the specified cache directory if not already there
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root=clip_cache_dir)
    return clip_model


def load_model(weights_path, device, num_classes):
    clip_model = load_clip_model(device)
    model = CLIPFineTunerWithAttention(clip_model, num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info("Model loaded successfully", extra={"weights_path": weights_path})
    return model

def preprocess_image(image):
    try:
        image = image.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as err:
        logger.error("Image preprocessing error", extra={"error": str(err)})
        st.error(f"Unsupported image type {type(image)}")

def log_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_usage_percent = ram_info.percent
    ram_usage_gb = ram_info.used / (1024 ** 3)
    gpu_memory_gb = memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    return cpu_percent, ram_usage_percent, ram_usage_gb, gpu_memory_gb

def similar_text_match(user_input, predicted_label, threshold=80):
    user_input = user_input.lower().strip()
    predicted_label = predicted_label.lower().strip()
    
    similarity_score = fuzz.ratio(user_input, predicted_label)
    
    return similarity_score >= threshold or user_input.lower() in predicted_label.lower()

def classify_image(uploaded_image, class_string, weights_path, subcategories):
    image = Image.open(uploaded_image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(subcategories)
    model = load_model(weights_path, device, num_classes)
    image_tensor = preprocess_image(image).to(device)
    predictions, inference_time = run_single_inference(model, image_tensor, device)
    predictions = torch.tensor(predictions)
    predicted_class = torch.argmax(predictions[0])
    predicted_label = subcategories[predicted_class]
    is_match = similar_text_match(class_string, predicted_label)
    cpu, ram_percent, ram_gb, gpu = log_resource_usage()
    logger.info("Inference completed", extra={
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": uploaded_image.name,
        "input_class": class_string,
        "predicted_class": predicted_label,
        "probability": predictions[0][predicted_class].item(),
        "inference_time": inference_time,
        "cpu_usage_percent": cpu,
        "ram_usage_percent":ram_percent,
        "ram_usage_gb": ram_gb,
        "gpu_usage_gb": gpu,
        "user_match": is_match 

    })
    
    return is_match, predicted_label, predictions[0][predicted_class].item(), inference_time

def run_single_inference(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    inference_time = time.time() - start_time
    return probabilities, inference_time

# Load subcategories
annotations_file = './final_aug_dataset/annotations.csv'
annotations = pd.read_csv(annotations_file)
subcategories = list(annotations['description'].unique())
weights_path = './model_registry/best_model.pth'

# Streamlit app code
st.markdown("<h1 style='text-align: center; color: darkblue;'>Image Classification</h1>", unsafe_allow_html=True)

# tab1 = st.tabs(["Classification"])

def reset_statistics(log_path='user_stats.csv'):
    if os.path.exists(log_path):
        os.remove(log_path)


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
class_input = st.text_input("Enter the expected class", help="E.g., 'Admiral company home appliances product'")
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
classify_button = st.button("Classify", help="Click to start classification", key='classify_button')
if classify_button and uploaded_image and class_input:
    image_name = uploaded_image.name
    is_match, predicted_class, probability, inference_time = classify_image(uploaded_image, class_input, weights_path, subcategories)
    cpu, ram_percent, ram_gb, gpu = log_resource_usage()
    if is_match:
        st.success(f"Classification successful! Predicted class: {predicted_class}. Matches your input.")
    else:
        st.error(f"Classification failed. Predicted class: {predicted_class}. Does not match your input.")
elif classify_button:
    st.warning("Please upload an image and enter the expected class.")


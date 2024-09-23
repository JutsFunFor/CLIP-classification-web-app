# utils.py
import yaml
import json
import os
import torch
from datetime import datetime
import clip

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_model_to_registry(model, registry_dir, model_name="model", version=1, metadata=None):
    if not os.path.exists(registry_dir):
        os.makedirs(registry_dir)
    version = version or len(os.listdir(registry_dir)) + 1
    model_file = os.path.join(registry_dir, f"{model_name}_v{version}.pth")
    torch.save(model.state_dict(), model_file)
    if metadata:
        metadata_file = os.path.join(registry_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
        else:
            metadata_dict = {}
        metadata_dict[f"{model_name}_v{version}"] = {"date": str(datetime.now()), "metrics": metadata}
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=4)

def load_model_from_registry(model, registry_dir, model_name="model", version=1):
    if version is None:
        version = max([int(f.split('_v')[-1].split('.pth')[0]) for f in os.listdir(registry_dir) if f.startswith(model_name)])
    model_file = os.path.join(registry_dir, f"{model_name}_v{version}.pth")
    model.load_state_dict(torch.load(model_file))
    return model


def load_clip_model(device):
    clip_cache_dir = os.getenv("CLIP_CACHE_DIR", "/app/.cache/clip")
    os.makedirs(clip_cache_dir, exist_ok=True)
    
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root=clip_cache_dir)
    return clip_model

def check_weights(path):
    return os.path.exists(path)
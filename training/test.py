# test.py
import torch
from utils import load_model_from_registry, load_config
from dataset import get_dataloaders
from train import evaluate_model
from model import CLIPFineTunerWithAttention
import clip

def main():
    config = load_config('config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, subcategories = get_dataloaders(config, config['registry_dir'], save_splits=False)
    
    clip_model, _ = clip.load("ViT-B/32", jit=False)
    model = CLIPFineTunerWithAttention(clip_model, len(subcategories)).to(device)

    model = load_model_from_registry(model, config['model_registry_dir'])
    accuracy, f1 = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy}, F1 Score: {f1}")

if __name__ == "__main__":
    main()

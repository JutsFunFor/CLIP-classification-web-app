import torch
from dataset import get_dataloaders
from model import CLIPFineTunerWithAttention
from train import train_and_validate
from utils import load_config, save_model_to_registry, load_clip_model
import os


def main():
    # Load configuration from YAML file
    config = load_config('config_docker.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data loaders for train, validation, and test, save_splits=True on the first run to save the split indices
    train_loader, val_loader, _, subcategories = get_dataloaders(config, config['registry_dir'], save_splits=True)
    
    clip_model = load_clip_model(device)
    # Initialize your fine-tuned model with attention
    model = CLIPFineTunerWithAttention(clip_model, len(subcategories)).to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.attention_head.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, config, device)

    # model_version = len([f for f in os.listdir(config['model_registry_dir']) if f.startswith('model')]) + 1
    # save_model_to_registry(model, config['model_registry_dir'], model_name="clip_model_attn", version=model_version)

if __name__ == "__main__":
    main()

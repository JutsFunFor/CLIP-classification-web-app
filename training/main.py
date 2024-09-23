import torch
from dataset import get_dataloaders
from model import CLIPFineTunerWithAttention
from train import train_and_validate
from utils import load_config, load_clip_model, check_weights


def start_training(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, subcategories = get_dataloaders(config, config['registry_dir'], save_splits=True)
    
    clip_model = load_clip_model(device)
    model = CLIPFineTunerWithAttention(clip_model, len(subcategories)).to(device)
    optimizer = torch.optim.AdamW(model.attention_head.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    train_and_validate(model, train_loader, val_loader, criterion, optimizer, config, device)

def main():
    weights_path = "../shared/model_registry/best_model.pth"
    is_weights = check_weights(weights_path)
    config = load_config('config_docker.yaml')

    if not is_weights or config['rewrite_model_weights']:
        start_training(config)  
   
if __name__ == "__main__":
    main()

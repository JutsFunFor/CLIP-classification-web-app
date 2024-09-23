import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pickle
import json
import torch
import numpy as np
import logging
import json_log_formatter

# JSON log formatter
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

logger = logging.getLogger("dataset_logger")
logger.setLevel(logging.INFO)
logger.addHandler(json_handler)
logger.propagate = False

class CustomClipDataset(Dataset):
    def __init__(self, annotations, img_dir, transform, subcategories):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.subcategories = subcategories

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.subcategories.index(self.annotations.iloc[idx, 1])
        return self.transform(image), label

def save_split_indices(train_indices, val_indices, test_indices, registry_dir, dataset, subcategories):
    if not os.path.exists(registry_dir):
        os.makedirs(registry_dir)
        logger.info("Created registry directory", extra={"directory": registry_dir})
    
    with open(os.path.join(registry_dir, 'train_indices.pkl'), 'wb') as f:
        pickle.dump(train_indices, f)
    with open(os.path.join(registry_dir, 'val_indices.pkl'), 'wb') as f:
        pickle.dump(val_indices, f)
    with open(os.path.join(registry_dir, 'test_indices.pkl'), 'wb') as f:
        pickle.dump(test_indices, f)
    
    train_labels = [dataset[i][1] for i in train_indices]
    val_labels = [dataset[i][1] for i in val_indices]
    test_labels = [dataset[i][1] for i in test_indices]
    
    train_class_distribution = {subcategories[int(k)]: int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))}
    val_class_distribution = {subcategories[int(k)]: int(v) for k, v in zip(*np.unique(val_labels, return_counts=True))}
    test_class_distribution = {subcategories[int(k)]: int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))}
    
    metadata = {
        "train_size": len(train_indices),
        "validation_size": len(val_indices),
        "test_size": len(test_indices),
        "train_class_distribution": train_class_distribution,
        "validation_class_distribution": val_class_distribution,
        "test_class_distribution": test_class_distribution
    }
    
    with open(os.path.join(registry_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("Saved split indices and metadata", extra={"registry_dir": registry_dir})

def load_split_indices(registry_dir):
    try:
        with open(os.path.join(registry_dir, 'train_indices.pkl'), 'rb') as f:
            train_indices = pickle.load(f)
        with open(os.path.join(registry_dir, 'val_indices.pkl'), 'rb') as f:
            val_indices = pickle.load(f)
        with open(os.path.join(registry_dir, 'test_indices.pkl'), 'rb') as f:
            test_indices = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Split indices not found in the registry directory. Ensure you have run the training process with `save_splits=True` first.")
    
    return train_indices, val_indices, test_indices

def random_split_indices(dataset_size, train_size, val_size):
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

def get_dataloaders(config, registry_dir, save_splits=False):
    annotations = pd.read_csv(config['annotations_path'])
    subcategories = annotations['description'].unique().tolist()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    dataset = CustomClipDataset(annotations, config['dataset_path'], transform, subcategories)

    if save_splits:
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        
        train_indices, val_indices, test_indices = random_split_indices(len(dataset), train_size, val_size)
        save_split_indices(train_indices, val_indices, test_indices, registry_dir, dataset, subcategories)
    else:
        train_indices, val_indices, test_indices = load_split_indices(registry_dir)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader, test_loader, subcategories

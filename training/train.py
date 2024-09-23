import os
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import json_log_formatter

# JSON log formatter setup
formatter = json_log_formatter.JSONFormatter()

# Ensure log directory exists
log_directory = "/app/logs"  # Change this path if necessary
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# File handler setup for logging
log_file_path = os.path.join(log_directory, "training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)

logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False  # Prevents duplicate logging if other handlers are added

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, config, device):
    writer = SummaryWriter(log_dir=config['log_dir'])
    best_val_f1 = 0.0
    early_stopping_counter = 0

    model_registry_dir = config['model_registry_dir']
    if not os.path.exists(model_registry_dir):
        os.makedirs(model_registry_dir)
        logger.info("Created model registry directory", extra={"directory": model_registry_dir})

    logger.info("Starting training", extra={"num_epochs": config['num_epochs'], "batch_size": config['batch_size']})

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        logger.info("Starting epoch", extra={"epoch": epoch + 1})

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}", leave=False)  # Only pbar in terminal
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_value'])
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_precision = precision_score(all_labels, all_predictions, average='macro')
        epoch_recall = recall_score(all_labels, all_predictions, average='macro')
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Precision/train', epoch_precision, epoch)
        writer.add_scalar('Recall/train', epoch_recall, epoch)
        writer.add_scalar('F1/train', epoch_f1, epoch)

        logger.info("Completed training for epoch", extra={
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1
        })

        val_accuracy, val_f1, val_precision, val_recall, per_class_f1 = evaluate_model(model, val_loader, device)
        
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        
        for i, f1 in enumerate(per_class_f1):
            writer.add_scalar(f'F1_per_class/class_{i}', f1, epoch)

        logger.info("Validation results", extra={
            "epoch": epoch + 1,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "per_class_f1": per_class_f1.tolist()
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(model_registry_dir, 'best_model.pth'))
            logger.info("New best model saved", extra={"epoch": epoch + 1, "best_val_f1": best_val_f1})
        else:
            early_stopping_counter += 1
            logger.info("No improvement in validation F1 score", extra={"epoch": epoch + 1, "early_stopping_counter": early_stopping_counter})

        if early_stopping_counter >= config['patience']:
            logger.info("Early stopping triggered", extra={"epoch": epoch + 1})
            break

    writer.close()
    logger.info("Training completed")


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    logger.info("Evaluation results", extra={
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "per_class_f1": per_class_f1.tolist()
    })
    
    return accuracy, f1, precision, recall, per_class_f1

#!/usr/bin/env python3
"""
SageMaker training script for financial fraud detection using TabFormer
This script is designed to run in a SageMaker training job with the NVIDIA financial fraud detection container
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    # TabFormer model hyperparameters
    parser.add_argument('--model-type', type=str, default='tabformer', choices=['tabformer', 'tabbert'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-attention-heads', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--class-weights', type=str, default='balanced', choices=['balanced', 'none'])
    parser.add_argument('--eval-metric', type=str, default='auc', choices=['auc', 'auprc'])
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--mixed-precision', type=bool, default=True)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)

    return parser.parse_args()


def load_data(data_dir):
    """Load training data from directory"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.parquet')]
    
    if not files:
        raise ValueError(f"No CSV or Parquet files found in {data_dir}")
    
    # Load the first file found
    file_path = os.path.join(data_dir, files[0])
    
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_parquet(file_path)
    
    return data


class FraudDataset(Dataset):
    """Custom dataset for financial fraud detection"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.LongTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data(data, args):
    """Preprocess financial transaction data for TabFormer"""
    logger.info("Preprocessing data for TabFormer...")

    # Handle target column - IEEE data uses 'isFraud'
    if 'isFraud' in data.columns and 'is_fraud' not in data.columns:
        data['is_fraud'] = data['isFraud']
        data = data.drop('isFraud', axis=1)
        logger.info("Mapped 'isFraud' to 'is_fraud'")
    elif 'is_fraud' not in data.columns:
        raise ValueError("Target column 'isFraud' or 'is_fraud' not found in data")

    # Log class distribution
    fraud_count = len(data[data['is_fraud'] == 1])
    legit_count = len(data[data['is_fraud'] == 0])
    total_count = len(data)
    fraud_rate = fraud_count / total_count

    logger.info(f"Dataset size: {total_count","} rows")
    logger.info(f"Legitimate transactions: {legit_count","} ({legit_count/total_count*100:.2f}%)")
    logger.info(f"Fraudulent transactions: {fraud_count","} ({fraud_rate*100:.4f}%)")

    # Separate features and target
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    # Handle missing values more carefully
    logger.info("Handling missing values...")

    # For categorical columns, fill with 'unknown'
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna('unknown')

    # For numerical columns, fill with median (more robust for fraud detection)
    numerical_cols = X.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())

    # Convert categorical variables efficiently
    logger.info("Encoding categorical variables...")
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes

    # Compute class weights for imbalanced dataset
    if args.class_weights == 'balanced':
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y.values)
        class_weights = torch.FloatTensor(class_weights)
        logger.info(f"Class weights: {class_weights}")
    else:
        class_weights = None

    return X, y, class_weights



def main():
    """Main training function"""
    args = parse_args()

    logger.info("Starting TabFormer training for fraud detection...")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Input arguments: {args}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Enable mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None

    # Load training data
    logger.info("Loading training data...")
    train_data = load_data(args.train)

    # Load and preprocess data
    logger.info("Preprocessing data...")
    X_train, y_train, class_weights = preprocess_data(train_data, args)

    # Load validation data
    logger.info("Loading validation data...")
    val_data = load_data(args.validation)
    X_val, y_val, _ = preprocess_data(val_data, args)

    # Create datasets and dataloaders
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    model = TabFormerModel(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout
    ).to(device)

    logger.info(f"Model architecture: {model}")
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_metric = 0
    patience_counter = 0
    training_history = []

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, class_weights, scaler)

        # Validate
        val_loss, val_acc, auc, auprc, precision, recall, f1, _, _ = validate(model, val_loader, criterion, device, class_weights)

        # Log metrics
        logger.info(f'Epoch {epoch+1:2d}/{args.epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'  AUC: {auc:.4f}, AUPRC: {auprc:.4f}')
        logger.info(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'auc': auc,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # Learning rate scheduling
        if args.eval_metric == 'auc':
            current_metric = auc
        else:
            current_metric = auprc

        scheduler.step(current_metric)

        # Early stopping
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'args': vars(args)
            }, best_model_path)
            logger.info(f'Saved best model with {args.eval_metric}: {best_metric:.4f}')
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    logger.info("Running final evaluation...")
    val_loss, val_acc, auc, auprc, precision, recall, f1, y_pred_proba, y_true = validate(model, val_loader, criterion, device, class_weights)

    logger.info("Final Validation Metrics:")
    logger.info(f"Accuracy: {val_acc:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")

    # Save metrics
    metrics = {
        'accuracy': val_acc / 100.0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'auprc': auprc,
        'best_epoch': checkpoint['epoch'],
        'training_time': datetime.now().isoformat()
    }

    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    history_path = os.path.join(args.output_data_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save final model
    model_path = os.path.join(args.model_dir, 'tabformer_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'args': vars(args),
        'metrics': metrics
    }, model_path)

    # Save model as joblib for compatibility
    model_config = {
        'model_type': args.model_type,
        'input_dim': input_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_attention_heads': args.num_attention_heads,
        'dropout': args.dropout,
        'metrics': metrics,
        'model_path': model_path
    }

    joblib_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model_config, joblib_path)

    logger.info("TabFormer training completed successfully!")
    logger.info(f"Best model saved to {best_model_path}")
    logger.info(f"Final model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()

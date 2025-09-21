#!/usr/bin/env python3
"""
SageMaker training script for financial fraud detection using XGBoost
This script is designed to run in a SageMaker training job
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Model hyperparameters
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--min-child-weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--colsample-bytree', type=float, default=1.0)
    parser.add_argument('--num-round', type=int, default=100)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval-metric', type=str, default='auc')
    
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


def preprocess_data(data):
    """Preprocess financial transaction data for fraud detection"""
    # Example preprocessing steps - adjust based on your actual data schema
    
    # Handle missing values
    data = data.fillna(0)
    
    # Convert categorical variables if present
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    
    # Assuming the target column is named 'is_fraud'
    if 'is_fraud' not in data.columns:
        raise ValueError("Target column 'is_fraud' not found in data")
    
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    return X, y


def main():
    """Main training function"""
    args = parse_args()
    
    print("Loading training data...")
    train_data = load_data(args.train)
    
    print("Preprocessing data...")
    X_train, y_train = preprocess_data(train_data)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set hyperparameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'objective': args.objective,
        'eval_metric': args.eval_metric,
        'verbosity': 1
    }
    
    print("Training XGBoost model...")
    evals = [(dtrain, 'train'), (dval, 'validation')]
    
    model = xgb.train(
        params,
        dtrain,
        args.num_round,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Make predictions
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"Validation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    # Save model
    model_path = os.path.join(args.model_dir, 'xgboost-model.json')
    model.save_model(model_path)
    
    # Also save as joblib for easier loading
    joblib_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, joblib_path)
    
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()

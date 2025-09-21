#!/usr/bin/env python3
"""
SageMaker preprocessing script for financial fraud detection data
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--input-data-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/processing/output')

    return parser.parse_args()

def load_data(data_dir):
    """Load raw data from input directory"""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') or f.endswith('.parquet')]

    if not files:
        raise ValueError(f"No CSV or Parquet files found in {data_dir}")

    # Load the first file found
    file_path = os.path.join(data_dir, files[0])

    logger.info(f"Loading data from {file_path}")

    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_parquet(file_path)

    logger.info(f"Data loaded. Shape: {data.shape}")
    return data

def preprocess_data(data):
    """Preprocess financial transaction data for fraud detection"""
    logger.info("Starting data preprocessing...")

    # Handle missing values
    data = data.fillna(0)

    # Convert categorical variables if present
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes

    # Assuming the target column is named 'is_fraud'
    if 'is_fraud' not in data.columns:
        raise ValueError("Target column 'is_fraud' not found in data")

    logger.info(f"Target column 'is_fraud' distribution:")
    logger.info(f"Class 0: {len(data[data['is_fraud'] == 0])}")
    logger.info(f"Class 1: {len(data[data['is_fraud'] == 1])}")

    # Separate features and target
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert back to DataFrames
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

    # Combine features and target
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)

    logger.info("Data preprocessing completed")
    logger.info(f"Training set shape: {train_data.shape}")
    logger.info(f"Validation set shape: {val_data.shape}")

    return train_data, val_data

def save_data(data, output_dir, filename):
    """Save preprocessed data to output directory"""
    output_path = os.path.join(output_dir, filename)
    data.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")

def main():
    """Main preprocessing function"""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_data_dir, exist_ok=True)

    try:
        # Load raw data
        raw_data = load_data(args.input_data_dir)

        # Preprocess data
        train_data, val_data = preprocess_data(raw_data)

        # Save preprocessed data
        save_data(train_data, args.output_data_dir, "train/train.csv")
        save_data(val_data, args.output_data_dir, "validation/validation.csv")

        # Save preprocessing metadata
        metadata = {
            "training_samples": len(train_data),
            "validation_samples": len(val_data),
            "features": list(train_data.columns[:-1]),  # Exclude target column
            "target_column": "is_fraud",
            "preprocessing_date": pd.Timestamp.now().isoformat()
        }

        metadata_path = os.path.join(args.output_data_dir, "preprocessing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Preprocessing completed successfully")
        logger.info(f"Metadata saved to {metadata_path}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == '__main__':
    main()

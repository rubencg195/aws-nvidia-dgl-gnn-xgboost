#!/usr/bin/env python3
"""
SageMaker Preprocessing Script for IEEE Fraud Detection
Prepares data for NVIDIA cuGraph GNN + XGBoost training
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def save_gnn_data(output_dir, edges_df, node_features_df, node_labels_df, tx_range, split_name='train'):
    """Save GNN data in NVIDIA container expected format"""
    base = Path(output_dir) / 'gnn' / (split_name if split_name != 'train' else 'train_gnn')
    edge_path = base / 'edges'
    node_path = base / 'nodes'
    edge_path.mkdir(parents=True, exist_ok=True)
    node_path.mkdir(parents=True, exist_ok=True)
    
    edges_df.to_csv(edge_path / 'node_to_node.csv', index=False)
    node_features_df.to_csv(node_path / 'node.csv', index=False)
    node_labels_df.to_csv(node_path / 'node_label.csv', index=False)
    
    if split_name == 'train':
        offset_info = {'start': int(tx_range[0]), 'end': int(tx_range[1])}
        with open(node_path / 'offset_range_of_training_node.json', 'w') as f:
            json.dump(offset_info, f, indent=4)
    
    print(f"✅ Saved GNN data for split={split_name} at {base}")


def save_xgb_data(output_dir, train_df, test_df, feature_cols, categorical_cols):
    """Save XGBoost data files"""
    xgb_path = Path(output_dir) / 'xgb'
    xgb_path.mkdir(parents=True, exist_ok=True)
    
    cols = list(feature_cols) + ['isFraud']
    train_df[cols].to_csv(xgb_path / 'training.csv', index=False)
    test_df[cols].to_csv(xgb_path / 'test.csv', index=False)
    
    cat_info = {
        'categorical_features': categorical_cols,
        'feature_names': list(feature_cols)
    }
    with open(xgb_path / 'feature_info.json', 'w') as f:
        json.dump(cat_info, f, indent=4)
    
    print(f"✅ Saved XGBoost data at {xgb_path}")


class GraphBuilder:
    """Build bipartite graph connecting transactions -> cards -> emails"""
    
    def __init__(self, df, card_col='card1', email_col='P_emaildomain', entity_vocab=None):
        self.df = df.copy().reset_index(drop=True)
        self.df['tx_id'] = self.df.index
        self.card_col = card_col
        self.email_col = email_col
        self.entity_vocab = entity_vocab

    def build_graph(self):
        if self.entity_vocab is None:
            # Build vocabulary from training data
            unique_cards = pd.Series(self.df[self.card_col].astype(str).unique())
            self.card_to_id = {card: idx for idx, card in enumerate(unique_cards)}
            
            unique_emails = pd.Series(self.df[self.email_col].astype(str).unique())
            self.email_to_id = {email: idx for idx, email in enumerate(unique_emails)}
            
            self.entity_vocab = {
                'card_to_id': self.card_to_id,
                'email_to_id': self.email_to_id
            }
        else:
            # Use existing vocabulary (for test set)
            self.card_to_id = self.entity_vocab['card_to_id']
            self.email_to_id = self.entity_vocab['email_to_id']
            
            # Filter out unseen entities
            before = len(self.df)
            self.df = self.df[
                self.df[self.card_col].astype(str).isin(self.card_to_id.keys()) &
                self.df[self.email_col].astype(str).isin(self.email_to_id.keys())
            ].reset_index(drop=True)
            self.df['tx_id'] = self.df.index
            after = len(self.df)
            print(f"Filtered unseen entities: before={before}, after={after} (dropped {before-after})")

        self.num_cards = len(self.card_to_id)
        self.num_emails = len(self.email_to_id)
        self.num_transactions = len(self.df)
        print(f"Graph nodes: {self.num_cards} cards, {self.num_emails} emails, {self.num_transactions} txs")
        return self

    def create_edges(self):
        """Create bipartite edges: card <-> transaction <-> email"""
        self.df['card_id'] = self.df[self.card_col].astype(str).map(self.card_to_id)
        self.df['email_id'] = self.df[self.email_col].astype(str).map(self.email_to_id)
        self.df = self.df.dropna(subset=['card_id', 'email_id']).copy()
        self.df['card_id'] = self.df['card_id'].astype(int)
        self.df['email_id'] = self.df['email_id'].astype(int)
        
        # Offset node IDs by type
        card_offset = 0
        email_offset = self.num_cards
        tx_offset = self.num_cards + self.num_emails
        
        edges = []
        for idx, row in self.df.iterrows():
            card_node = card_offset + row['card_id']
            email_node = email_offset + row['email_id']
            tx_node = tx_offset + row['tx_id']
            
            # Bidirectional edges
            edges.extend([
                [card_node, tx_node],
                [tx_node, email_node],
                [tx_node, card_node],
                [email_node, tx_node]
            ])
        
        return pd.DataFrame(edges, columns=['src', 'dst'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-transaction', type=str, default='/opt/ml/processing/input/train_transaction.csv')
    parser.add_argument('--train-identity', type=str, default='/opt/ml/processing/input/train_identity.csv')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size for debugging')
    args = parser.parse_args()
    
    print("=" * 80)
    print("IEEE Fraud Detection - Preprocessing for NVIDIA cuGraph GNN + XGBoost")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading data from {args.train_transaction}...")
    train_transaction = pd.read_csv(args.train_transaction)
    
    if os.path.exists(args.train_identity):
        print(f"📂 Loading identity data from {args.train_identity}...")
        train_identity = pd.read_csv(args.train_identity)
    else:
        print("⚠️  No identity file found, proceeding without it")
        train_identity = None
    
    # Optional sampling for debugging
    if args.sample_size:
        print(f"⚠️  Sampling {args.sample_size} transactions for debugging")
        train_transaction = train_transaction.head(args.sample_size)
        if train_identity is not None:
            train_identity = train_identity[
                train_identity['TransactionID'].isin(train_transaction['TransactionID'])
            ]
    
    # Merge
    print("\n🔗 Merging transaction and identity data...")
    if train_identity is not None:
        train_merged = train_transaction.merge(train_identity, on="TransactionID", how="left")
    else:
        train_merged = train_transaction.copy()
    
    print(f"Merged dataset shape: {train_merged.shape}")
    
    # Replace placeholder values
    print("\n🧹 Cleaning placeholder values...")
    train_merged.replace(["unknown", "Unknown", "UNKN"], np.nan, inplace=True)
    
    # Identify features
    print("\n🔍 Identifying categorical and numerical features...")
    categorical_features = [
        c for c in train_merged.columns
        if train_merged[c].dtype == "object" or train_merged[c].nunique() < 20
    ]
    numerical_features = [
        c for c in train_merged.columns
        if c not in categorical_features + ["isFraud"]
    ]
    print(f"Found {len(categorical_features)} categorical, {len(numerical_features)} numerical features")
    
    # Fill missing values
    print("\n🔧 Filling missing values...")
    for c in categorical_features:
        train_merged[c] = train_merged[c].fillna("missing")
    for c in numerical_features:
        train_merged[c] = train_merged[c].fillna(train_merged[c].median())
    
    # Encode categoricals
    print("\n🔢 Encoding categorical features...")
    encoders = {}
    for c in categorical_features:
        le = LabelEncoder()
        train_merged[c] = le.fit_transform(train_merged[c].astype(str))
        encoders[c] = le
    
    # Scale numericals
    print("\n📊 Scaling numerical features...")
    scaler = StandardScaler()
    train_merged[numerical_features] = scaler.fit_transform(
        train_merged[numerical_features].astype(float)
    )
    
    print("✅ Preprocessing complete on full dataset")
    
    # Time-based split (70/30)
    print("\n✂️  Performing time-based split (70% train, 30% test)...")
    train_merged = train_merged.sort_values('TransactionDT').reset_index(drop=True)
    n = len(train_merged)
    train_end = int(n * 0.7)
    
    train_df = train_merged.iloc[:train_end].reset_index(drop=True)
    test_df = train_merged.iloc[train_end:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} transactions (fraud rate: {train_df['isFraud'].mean():.4f})")
    print(f"Test:  {len(test_df)} transactions (fraud rate: {test_df['isFraud'].mean():.4f})")
    
    # Build graph data
    feature_cols = categorical_features + numerical_features
    print(f"\n🕸️  Building graph with {len(feature_cols)} features...")
    
    print("Building train graph...")
    train_graph = GraphBuilder(train_df).build_graph()
    train_edges = train_graph.create_edges()
    save_gnn_data(
        args.output_path, 
        train_edges, 
        train_df[feature_cols], 
        train_df[['isFraud']], 
        (0, len(train_df)), 
        split_name='train'
    )
    
    print("\nBuilding test graph...")
    test_graph = GraphBuilder(test_df, entity_vocab=train_graph.entity_vocab).build_graph()
    test_edges = test_graph.create_edges()
    save_gnn_data(
        args.output_path, 
        test_edges, 
        test_df[feature_cols], 
        test_df[['isFraud']], 
        (len(train_df), len(train_df)+len(test_df)), 
        split_name='test'
    )
    
    # Save XGBoost data
    print("\n💾 Saving XGBoost training/test files...")
    save_xgb_data(args.output_path, train_df, test_df, feature_cols, categorical_features)
    
    # Create training config
    print("\n⚙️  Creating training configuration...")
    fraud_count = int(train_df['isFraud'].sum())
    non_fraud_count = int(len(train_df) - fraud_count)
    scale_pos_weight = float(non_fraud_count) / max(1.0, float(fraud_count))
    
    config = {
        "model_type": "graphsage_xgboost",
        "data_dir": "/opt/ml/input/data/training",
        "output_dir": "/opt/ml/model",
        "gnn_config": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 10
        },
        "xgb_config": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "scale_pos_weight": scale_pos_weight
        }
    }
    
    config_dir = Path(args.output_path) / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Saved config with scale_pos_weight={scale_pos_weight:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ All preprocessing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()


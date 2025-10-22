#!/usr/bin/env python3
"""
Generate training configuration for NVIDIA financial fraud training container
"""

import json
import sys
import os

def generate_training_config():
    """Generate training configuration for GraphSAGE + XGBoost model"""
    
    training_config = {
        "paths": {
            "data_dir": "/opt/ml/input/data/gnn",
            "output_dir": "/opt/ml/model"
        },
        "models": [
            {
                "kind": "GraphSAGE_XGBoost",
                "gpu": "single",
                "hyperparameters": {
                    "gnn": {
                        "hidden_channels": 32,
                        "n_hops": 2,
                        "dropout_prob": 0.2,
                        "batch_size": 1024,
                        "fan_out": 32,
                        "num_epochs": 20
                    },
                    "xgb": {
                        "max_depth": 8,
                        "learning_rate": 0.1,
                        "num_parallel_tree": 1,
                        "num_boost_round": 1000,
                        "gamma": 1.0
                    }
                }
            }
        ]
    }
    
    return training_config

if __name__ == "__main__":
    config = generate_training_config()
    
    # Output to stdout if no argument provided, otherwise write to file
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration written to {output_file}")
    else:
        print(json.dumps(config, indent=2))

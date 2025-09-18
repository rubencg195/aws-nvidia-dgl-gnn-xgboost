# Graph Neural Network Demo - Financial Fraud Detection

A demonstration project for financial fraud detection using machine learning models deployed to AWS SageMaker via OpenTofu infrastructure as code.

## Project Structure

```
graph-neural-network-demo/
├── devops/
│   └── main.tf              # OpenTofu configuration for AWS infrastructure
├── training-script/
│   └── train.py             # SageMaker training script for fraud detection
└── README.md
```

## Overview

This project demonstrates an end-to-end machine learning pipeline for financial fraud detection:

- **Infrastructure as Code**: OpenTofu (Terraform-compatible) configuration for AWS resources
- **Model Training**: XGBoost-based fraud detection model optimized for financial transaction data
- **SageMaker Integration**: Ready for deployment to AWS SageMaker training jobs

## Features

- **Financial Fraud Detection**: Specialized for detecting fraudulent financial transactions
- **XGBoost Implementation**: High-performance gradient boosting model
- **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1 score, and AUC
- **OpenTofu Deployment**: Infrastructure as code for reproducible deployments
- **SageMaker Compatible**: Designed to run in AWS SageMaker training environments

## Training Script (`training-script/train.py`)

The training script includes:

- Data loading from CSV/Parquet files
- Preprocessing for financial transaction data
- XGBoost model training with hyperparameter configuration
- Validation metrics calculation and tracking
- Model serialization in multiple formats (XGBoost native + joblib)

### Expected Data Schema

The script expects financial transaction data with:
- Various feature columns (transaction amount, location, time, etc.)
- A binary target column named `is_fraud` (1 = fraud, 0 = legitimate)

## Infrastructure (`devops/main.tf`)

OpenTofu configuration featuring:
- AWS provider setup with default tags
- Common infrastructure patterns for SageMaker deployment
- Region configuration (currently us-east-1)
- Required providers for AWS and null resources

## Setup Instructions

### Prerequisites

- Python 3.8+ with required packages (see requirements.txt)
- AWS CLI configured with appropriate permissions
- OpenTofu or Terraform installed

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd graph-neural-network-demo
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize OpenTofu:
```bash
cd devops
tofu init
```

### Usage

1. **Prepare your financial transaction data** in CSV or Parquet format
2. **Configure AWS credentials** for SageMaker access
3. **Deploy infrastructure** with OpenTofu:
```bash
cd devops
tofu plan
tofu apply -auto-approve
```

4. **Run training jobs** through SageMaker or locally for testing

## Next Steps

- Add Graph Neural Network implementation for enhanced fraud detection
- Expand OpenTofu configuration for complete SageMaker pipeline
- Implement CI/CD pipelines for automated training and deployment
- Add monitoring and alerting for model performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.
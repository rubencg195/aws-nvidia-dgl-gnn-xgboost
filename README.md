# Financial Fraud Detection with GNN + XGBoost on AWS SageMaker

An end-to-end machine learning pipeline for financial fraud detection using Graph Neural Networks (DGL) combined with XGBoost, deployed on AWS SageMaker. This project leverages NVIDIA's cuGraph financial fraud training container to train state-of-the-art models on the IEEE-CIS Fraud Detection dataset, with full infrastructure automation via OpenTofu.

## 🎯 Project Overview

This project demonstrates a production-ready pipeline that:
- **Preprocesses** the IEEE-CIS Fraud Detection dataset using SageMaker Processing Jobs
- **Trains** a hybrid GNN (Graph Neural Network) + XGBoost model using NVIDIA's optimized containers on GPU instances
- **Deploys** the trained model for inference
- **Automates** the entire workflow using OpenTofu (Terraform) infrastructure as code

### Key Technologies
- **NVIDIA cuGraph**: GPU-accelerated graph analytics and GNN training
- **DGL (Deep Graph Library)**: Graph neural network framework
- **XGBoost**: Gradient boosting for tabular features
- **AWS SageMaker**: Managed ML platform for training and deployment
- **OpenTofu**: Infrastructure as code for reproducible deployments

## 📊 Dataset

The project uses the **IEEE-CIS Fraud Detection** dataset, which includes:
- **590,540 transactions** with 434 features after preprocessing
- Transaction data (`train_transaction.csv`) with payment information
- Identity data (`train_identity.csv`) with device and network information
- Highly imbalanced (~3.5% fraud rate)

## 🏗️ Architecture

### Pipeline Stages

#### 1. **Data Preprocessing** (SageMaker Processing Job)
The preprocessing stage (`notebooks/nvidia/preprocessing.ipynb`) performs:
- Merges transaction and identity data
- Feature engineering: categorical encoding, numerical scaling
- Graph construction: Creates bipartite graph connecting transactions → cards → emails
- Time-based train/test split (70/30)
- Outputs data in NVIDIA container format:
  ```
  processed/ieee-fraud-detection/
  ├── gnn/
  │   ├── train_gnn/
  │   │   ├── edges/node_to_node.csv
  │   │   └── nodes/
  │   │       ├── node.csv (features)
  │   │       ├── node_label.csv (fraud labels)
  │   │       └── offset_range_of_training_node.json
  │   └── test/
  │       ├── edges/node_to_node.csv
  │       └── nodes/
  └── xgb/
      ├── training.csv
      ├── test.csv
      └── feature_info.json
  ```

#### 2. **Container Management** (`docker-upload.tf`)
- Creates AWS ECR repository for the NVIDIA container
- Pulls `nvcr.io/nvidia/cugraph/financial-fraud-training:1.0.1` (33GB GPU-optimized image)
- Authenticates with NGC (NVIDIA GPU Cloud) using credentials from `nvidia_credentials.json`
- Tags and pushes to ECR for SageMaker training jobs
- Implements idempotent checks to skip if image already exists

#### 3. **Model Training** (SageMaker Training Job)
The training stage (`notebooks/nvidia/training-job.ipynb`) configures:
- **Model Architecture**: GraphSAGE GNN + XGBoost ensemble
- **GPU Instance**: ml.g5.xlarge or similar (NVIDIA A10G)
- **Training Config** (`training_config.json`):
  ```json
  {
    "model_type": "graphsage_xgboost",
    "data_dir": "/opt/ml/input/data/training",
    "output_dir": "/opt/ml/model",
    "gnn_config": {
      "hidden_dim": 128,
      "num_layers": 3,
      "dropout": 0.2,
      "learning_rate": 0.001
    },
    "xgb_config": {
      "max_depth": 6,
      "learning_rate": 0.1,
      "n_estimators": 100,
      "scale_pos_weight": 27.43
    }
  }
  ```

#### 4. **Deployment** (Future)
- SageMaker Endpoint for real-time inference
- Batch transform for bulk predictions

## 🚀 Getting Started

### Prerequisites

1. **AWS Account** with SageMaker access
2. **NVIDIA NGC Account** for container access ([sign up](https://ngc.nvidia.com/))
3. **OpenTofu** (or Terraform) installed
4. **Docker** installed locally
5. **AWS CLI** configured with credentials

### Setup Instructions

#### 1. Configure NVIDIA Credentials
Create `nvidia_credentials.json` at the project root (excluded from git):
```json
{
  "ngc_api_key": "YOUR_NGC_API_KEY_HERE"
}
```

Get your NGC API key from: https://ngc.nvidia.com/setup/api-key

#### 2. Upload Container to ECR
```bash
# Initialize OpenTofu
tofu init -upgrade

# Deploy ECR repository and upload NVIDIA container
tofu apply -auto-approve
```

This will:
- Create the ECR repository `financial-fraud-training`
- Pull the 33GB NVIDIA container from NGC (first run only)
- Push to your ECR registry

#### 3. Prepare Data
Follow the preprocessing notebook to:
1. Upload IEEE dataset to S3
2. Run preprocessing to create graph structure
3. Upload processed data to S3

```bash
# Run preprocessing notebook or script
jupyter notebook notebooks/nvidia/preprocessing.ipynb
```

#### 4. Train the Model
```bash
# Run training job notebook
jupyter notebook notebooks/nvidia/training-job.ipynb
```

The training job will:
- Spin up a GPU instance (ml.g5.xlarge)
- Load processed data from S3
- Train GraphSAGE GNN to learn transaction embeddings
- Train XGBoost on combined GNN embeddings + tabular features
- Save model artifacts to S3

## 📁 Project Structure

```
graph-neural-network-demo/
├── data/                           # Raw IEEE dataset (gitignored)
│   ├── train_identity.csv
│   └── train_transaction.csv
├── docker-upload.tf                # OpenTofu config for NVIDIA container
├── nvidia_credentials.json         # NGC API key (gitignored)
├── notebooks/
│   └── nvidia/
│       ├── preprocessing.ipynb     # Data preprocessing for GNN + XGBoost
│       ├── download-upload-ecr.ipynb  # Container management notebook
│       └── training-job.ipynb      # SageMaker training job launcher
├── scripts/
│   └── training/
│       └── train.py                # Custom training script (if needed)
├── provider.tf                     # AWS provider configuration
├── s3.tf                           # S3 bucket for data/models
├── vpc.tf                          # VPC configuration for SageMaker
└── README.md
```

## 🔧 Infrastructure Components

### OpenTofu Resources
- **ECR Repository**: Stores NVIDIA training container
- **S3 Buckets**: Data storage and model artifacts
- **VPC**: Network isolation for SageMaker jobs
- **IAM Roles**: SageMaker execution permissions

### Key Files
- `docker-upload.tf`: Automates container pull/push from NGC to ECR
- `nvidia_credentials.json`: Secure storage for NGC API key (in .gitignore)
- `locals.tf`: Common tags and configuration variables

## 📊 Model Performance

The hybrid GNN + XGBoost approach provides:
- **Graph Learning**: GraphSAGE captures relationship patterns between transactions, cards, and emails
- **Tabular Features**: XGBoost leverages traditional fraud indicators
- **Ensemble Benefits**: Combines structural and feature-based signals

Expected performance on IEEE dataset:
- **AUC-ROC**: ~0.92-0.95
- **F1-Score**: ~0.70-0.75 (with threshold tuning)

## 🔒 Security Notes

- `nvidia_credentials.json` is in `.gitignore` - never commit credentials
- Use IAM roles for SageMaker instead of hardcoded AWS keys
- ECR repositories use AES256 encryption
- Training jobs run in private VPC subnets

## 🐛 Troubleshooting

### Container Pull Issues
```bash
# Check if image exists in ECR
aws ecr describe-images --repository-name financial-fraud-training --region us-east-1

# Force re-pull by deleting ECR image
aws ecr batch-delete-image --repository-name financial-fraud-training --image-ids imageTag=1.0.1
tofu apply -auto-approve
```

### Training Job Failures
- Check CloudWatch logs: `/aws/sagemaker/TrainingJobs`
- Verify S3 paths in training config
- Ensure GPU instance quota is available
- Validate preprocessed data structure

## 📚 References

- [NVIDIA cuGraph Documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [DGL Documentation](https://docs.dgl.ai/)
- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
- [AWS SageMaker Training Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html)

## 📝 License

This project is for demonstration purposes. Check individual component licenses:
- NVIDIA containers: [NGC License](https://ngc.nvidia.com/legal)
- IEEE dataset: Kaggle competition terms

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test changes with `tofu plan`
4. Submit a pull request

---

**Note**: This pipeline requires GPU instances for training. Typical training time: 15-30 minutes on ml.g5.xlarge for the full IEEE dataset.
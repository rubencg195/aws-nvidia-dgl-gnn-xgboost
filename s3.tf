# S3 Bucket for training input data
resource "aws_s3_bucket" "training_input" {
  bucket = "aws-nvidia-dgl-gnn-xgboost-training-input-us-east-1"

  tags = {
    Project     = "aws-nvidia-dgl-gnn-xgboost"
    Environment = "dev"
    ManagedBy   = "opentofu"
    Name        = "Training Input Bucket"
  }
}

resource "aws_s3_bucket_versioning" "training_input" {
  bucket = aws_s3_bucket.training_input.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket for training output (models, artifacts)
resource "aws_s3_bucket" "training_output" {
  bucket = "aws-nvidia-dgl-gnn-xgboost-training-output-us-east-1"

  tags = {
    Project     = "aws-nvidia-dgl-gnn-xgboost"
    Environment = "dev"
    ManagedBy   = "opentofu"
    Name        = "Training Output Bucket"
  }
}

resource "aws_s3_bucket_versioning" "training_output" {
  bucket = aws_s3_bucket.training_output.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Upload training data files to S3 input bucket
resource "aws_s3_object" "train_identity_data" {
  bucket = aws_s3_bucket.training_input.id
  key    = "raw-data/train_identity.csv"
  source = "${path.module}/data/train_identity.csv"

  depends_on = [
    aws_s3_bucket.training_input
  ]

  tags = {
    Project     = "aws-nvidia-dgl-gnn-xgboost"
    Environment = "dev"
    ManagedBy   = "opentofu"
    Name        = "Training Identity Data"
  }
}

resource "aws_s3_object" "train_transaction_data" {
  bucket = aws_s3_bucket.training_input.id
  key    = "raw-data/train_transaction.csv"
  source = "${path.module}/data/train_transaction.csv"

  depends_on = [
    aws_s3_bucket.training_input
  ]

  tags = {
    Project     = "aws-nvidia-dgl-gnn-xgboost"
    Environment = "dev"
    ManagedBy   = "opentofu"
    Name        = "Training Transaction Data"
  }
}

# Upload training script to S3 input bucket
resource "aws_s3_object" "training_script" {
  bucket = aws_s3_bucket.training_input.id
  key    = "code/train.py"
  source = "${path.module}/scripts/training/train.py"

  depends_on = [
    aws_s3_bucket.training_input
  ]

  tags = {
    Project     = "aws-nvidia-dgl-gnn-xgboost"
    Environment = "dev"
    ManagedBy   = "opentofu"
    Name        = "Training Script"
  }
}
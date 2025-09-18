# S3 Bucket for training input data
resource "aws_s3_bucket" "training_input" {
  bucket = "graph-neural-network-demo-training-input-us-east-1"

  tags = {
    Project     = "graph-neural-network-demo"
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
  bucket = "graph-neural-network-demo-training-output-us-east-1"

  tags = {
    Project     = "graph-neural-network-demo"
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

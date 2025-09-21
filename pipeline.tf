# IAM Role for SageMaker pipeline execution
resource "aws_iam_role" "sagemaker_pipeline" {
  name = "graph-neural-network-demo-pipeline-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Project     = "graph-neural-network-demo"
    Environment = "dev"
    ManagedBy   = "opentofu"
  }
}

# IAM Role for SageMaker job execution
resource "aws_iam_role" "sagemaker_execution" {
  name = "graph-neural-network-demo-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Project     = "graph-neural-network-demo"
    Environment = "dev"
    ManagedBy   = "opentofu"
  }
}

# IAM Policy for SageMaker pipeline execution
resource "aws_iam_role_policy" "sagemaker_pipeline" {
  name = "graph-neural-network-demo-pipeline-policy"
  role = aws_iam_role.sagemaker_pipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "sagemaker:CreatePipeline",
          "sagemaker:DescribePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:SendPipelineExecutionStepSuccess",
          "sagemaker:SendPipelineExecutionStepFailure"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:StopProcessingJob",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.training_input.arn,
          "${aws_s3_bucket.training_input.arn}/*",
          aws_s3_bucket.training_output.arn,
          "${aws_s3_bucket.training_output.arn}/*"
        ]
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# IAM Policy for SageMaker execution role
resource "aws_iam_role_policy" "sagemaker_execution" {
  name = "graph-neural-network-demo-sagemaker-execution-policy"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.training_input.arn,
          "${aws_s3_bucket.training_input.arn}/*",
          aws_s3_bucket.training_output.arn,
          "${aws_s3_bucket.training_output.arn}/*"
        ]
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# SageMaker Pipeline Definition
resource "aws_sagemaker_pipeline" "graph_neural_network" {
  pipeline_name         = "graph-neural-network-demo-pipeline"
  pipeline_display_name = "Graph-Neural-Network-Demo-Pipeline"
  role_arn              = aws_iam_role.sagemaker_pipeline.arn

  depends_on = [
    aws_s3_object.train_identity_data,
    aws_s3_object.train_transaction_data,
    aws_s3_object.training_script
  ]

  pipeline_definition = jsonencode({
    Version = "2020-12-01"
    Steps = [
      # Preprocessing Step using ScriptProcessor
      {
        Name = "PreprocessingStep"
        Type = "Processing"
        Arguments = {
          ProcessingInputs = [
            {
              InputName = "input"
              AppManaged = false
              S3Input = {
                S3Uri = "s3://${aws_s3_bucket.training_input.bucket}/raw-data/"
                LocalPath = "/opt/ml/processing/input"
                S3DataType = "S3Prefix"
                S3InputMode = "File"
                S3DataDistributionType = "FullyReplicated"
              }
            }
          ]
          ProcessingOutputConfig = {
            Outputs = [
              {
                OutputName = "preprocessed_data"
                AppManaged = false
                S3Output = {
                  S3Uri = "s3://${aws_s3_bucket.training_output.bucket}/preprocessing/"
                  LocalPath = "/opt/ml/processing/output"
                  S3UploadMode = "EndOfJob"
                }
              }
            ]
          }
          ProcessingResources = {
            ClusterConfig = {
              InstanceCount = 1
              InstanceType = "ml.m5.xlarge"
              VolumeSizeInGB = 30
            }
          }
          AppSpecification = {
            ImageUri = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
            ContainerEntrypoint = ["python3", "/opt/ml/processing/code/preprocessing.py"]
          }
          Environment = {
            AWS_DEFAULT_REGION = "us-east-1"
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
        }
      },
      # Training Step
      {
        Name = "TrainingStep"
        Type = "Training"
        DependsOn = ["PreprocessingStep"]
        Arguments = {
          AlgorithmSpecification = {
            TrainingImage = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1"
            TrainingInputMode = "File"
          }
          InputDataConfig = [
            {
              ChannelName = "train"
              DataSource = {
                S3DataSource = {
                  S3DataType = "S3Prefix"
                  S3Uri = "s3://${aws_s3_bucket.training_output.bucket}/preprocessing/train/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
              ContentType = "text/csv"
            },
            {
              ChannelName = "validation"
              DataSource = {
                S3DataSource = {
                  S3DataType = "S3Prefix"
                  S3Uri = "s3://${aws_s3_bucket.training_output.bucket}/preprocessing/validation/"
                  S3DataDistributionType = "FullyReplicated"
                }
              }
              ContentType = "text/csv"
            }
          ]
          OutputDataConfig = {
            S3OutputPath = "s3://${aws_s3_bucket.training_output.bucket}/models/"
          }
          ResourceConfig = {
            InstanceCount = 1
            InstanceType = "ml.m5.xlarge"
            VolumeSizeInGB = 30
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 86400
          }
          HyperParameters = {
            max_depth = "6"
            eta = "0.3"
            objective = "binary:logistic"
            num_round = "100"
            eval_metric = "auc"
          }
          Environment = {
            SAGEMAKER_PROGRAM = "train.py"
            SAGEMAKER_SUBMIT_DIRECTORY = "s3://${aws_s3_bucket.training_input.bucket}/code/train.py"
          }
          RoleArn = aws_iam_role.sagemaker_execution.arn
        }
      }
    ]
  })

  tags = {
    Project     = "graph-neural-network-demo"
    Environment = "dev"
    ManagedBy   = "opentofu"
  }
}

# Create preprocessing script
resource "local_file" "preprocessing_script" {
  content = <<EOF
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
EOF
  filename = "${path.module}/scripts/preprocessing/preprocessing.py"
}

# Update Lambda function IAM policy to include pipeline execution permissions
resource "aws_iam_role_policy_attachment" "sagemaker_pipeline_access" {
  role       = aws_iam_role.sagemaker_job_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}


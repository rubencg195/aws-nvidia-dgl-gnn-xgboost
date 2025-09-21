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
              InstanceType = "ml.m5.4xlarge"
              VolumeSizeInGB = 100
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
            TrainingImage = "763104351884.dkr.ecr.us-east-1.amazonaws.com/financial-fraud-training:latest"
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
            InstanceType = "ml.m5.4xlarge"
            VolumeSizeInGB = 100
          }
          StoppingCondition = {
            MaxRuntimeInSeconds = 86400
          }
          HyperParameters = {
            model_type = "tabformer"
            epochs = "50"
            batch_size = "1024"
            learning_rate = "0.001"
            hidden_size = "768"
            num_layers = "6"
            num_attention_heads = "12"
            dropout = "0.1"
            class_weights = "balanced"
            eval_metric = "auc"
            early_stopping_patience = "10"
            mixed_precision = "true"
            gradient_accumulation_steps = "1"
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
    """Load raw data from input directory or file"""
    logger.info(f"Input data directory: {data_dir}")

    # Handle case where data_dir might be a file or directory
    if os.path.isfile(data_dir):
        # Direct file path (S3Object case)
        file_path = data_dir
        logger.info(f"Loading data from file: {file_path}")
    else:
        # Directory containing files (S3Prefix case)
        logger.info("Listing all files in input directory...")
        try:
            all_files = os.listdir(data_dir)
            logger.info(f"Files in input directory: {all_files}")

            # Check if there's a manifest file (SageMaker S3Prefix creates this)
            manifest_files = [f for f in all_files if f.endswith('.json')]
            if manifest_files:
                logger.info(f"Found manifest files: {manifest_files}")
                manifest_path = os.path.join(data_dir, manifest_files[0])
                try:
                    with open(manifest_path, 'r') as f:
                        import json
                        manifest = json.load(f)
                        logger.info(f"Manifest content: {json.dumps(manifest, indent=2)}")

                    # Get the first data file from manifest
                    if 'entries' in manifest and len(manifest['entries']) > 0:
                        first_entry = manifest['entries'][0]
                        if 'uri' in first_entry:
                            # Extract file path from S3 URI
                            uri = first_entry['uri']
                            file_name = uri.split('/')[-1]
                            file_path = os.path.join(data_dir, file_name)
                            logger.info(f"Loading data from manifest file: {file_path}")
                        else:
                            raise ValueError("Manifest file does not contain expected format")
                    else:
                        raise ValueError("Manifest file is empty or invalid")
                except Exception as e:
                    logger.error(f"Error reading manifest file: {e}")
                    raise
            else:
                logger.info("No manifest files found")

            # Check for CSV files
            csv_files = [f for f in all_files if f.endswith('.csv')]
            if csv_files:
                logger.info(f"Found CSV files: {csv_files}")
                file_path = os.path.join(data_dir, csv_files[0])
                logger.info(f"Loading data from CSV file: {file_path}")
            else:
                logger.info("No CSV files found")
                raise ValueError(f"No CSV files found in {data_dir}")

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            raise

    # Handle different file types
    if file_path.endswith('.csv'):
        logger.info("Loading CSV file...")
        data = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        logger.info("Loading Parquet file...")
        data = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    return data

def preprocess_data(data):
    """Preprocess financial transaction data for fraud detection"""
    logger.info("Starting data preprocessing...")

    # Handle target column - IEEE data uses 'isFraud' (capital F)
    if 'isFraud' in data.columns:
        data = data.rename(columns={'isFraud': 'is_fraud'})
        logger.info("Mapped 'isFraud' to 'is_fraud'")
    elif 'is_fraud' not in data.columns:
        logger.error("Target column 'is_fraud' not found in data")
        raise ValueError("Target column 'is_fraud' not found in data")

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)[:10]}...")

    # Log class distribution
    if 'is_fraud' in data.columns:
        fraud_count = data['is_fraud'].sum()
        total_count = len(data)
        logger.info(f"Fraud rate: {fraud_count}/{total_count} ({fraud_count/total_count*100:.4f}%)")

    # Simple train/validation split (use first 80% for training, last 20% for validation)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]

    logger.info("Data preprocessing completed successfully")
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
            "preprocessing_date": str(pd.Timestamp.now())
        }

        metadata_path = os.path.join(args.output_data_dir, "preprocessing_metadata.json")
        with open(metadata_path, 'w') as f:
            import json
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


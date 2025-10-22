# SageMaker Training Job Infrastructure using OpenTofu
# Automates deployment of GraphSAGE + XGBoost model training on AWS SageMaker

# Load NGC credentials from JSON file
locals {
  nvidia_credentials = try(jsondecode(file("${path.module}/nvidia_credentials.json")), { ngc_api_key = "" })
  ngc_api_key        = local.nvidia_credentials.ngc_api_key
}

resource "aws_iam_role" "sagemaker_training" {
  name = "${local.project_name}-sagemaker-training-role"
  
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
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_training_full_access" {
  role       = aws_iam_role.sagemaker_training.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_training_s3_access" {
  role       = aws_iam_role.sagemaker_training.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_training_ecr_access" {
  role       = aws_iam_role.sagemaker_training.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Upload wrapper script to S3
resource "aws_s3_object" "training_wrapper_script" {
  bucket = aws_s3_bucket.training_input.id
  key    = "code/training/wrapper.sh"
  source = "${path.module}/scripts/training/wrapper.sh"
  depends_on = [aws_s3_bucket.training_input]
  
  tags = merge(local.common_tags, { Name = "Training Wrapper Script" })
}

# Upload training configuration generator to S3
resource "aws_s3_object" "training_config_generator" {
  bucket = aws_s3_bucket.training_input.id
  key    = "code/training/generate_config.py"
  source = "${path.module}/scripts/training/generate_config.py"
  depends_on = [aws_s3_bucket.training_input]
  
  tags = merge(local.common_tags, { Name = "Training Config Generator" })
}

# Main training job orchestration
resource "null_resource" "run_training_job" {
  depends_on = [
    aws_iam_role.sagemaker_training,
    aws_iam_role_policy_attachment.sagemaker_training_full_access,
    aws_iam_role_policy_attachment.sagemaker_training_s3_access,
    aws_iam_role_policy_attachment.sagemaker_training_ecr_access,
    aws_s3_object.training_wrapper_script,
    aws_s3_object.training_config_generator,
    null_resource.run_preprocessing_job
  ]
  
  triggers = {
    wrapper_hash        = filesha256("${path.module}/scripts/training/wrapper.sh")
    config_gen_hash     = filesha256("${path.module}/scripts/training/generate_config.py")
    role_arn            = aws_iam_role.sagemaker_training.arn
    preprocessing_id    = null_resource.run_preprocessing_job.id
  }
  
  provisioner "local-exec" {
    interpreter = ["bash", "-lc"]
    command = <<EOT
set -euo pipefail
set +H

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${data.aws_region.current.name}
ROLE_ARN="${aws_iam_role.sagemaker_training.arn}"
INPUT_BUCKET="${aws_s3_bucket.training_input.bucket}"
OUTPUT_BUCKET="${aws_s3_bucket.training_output.bucket}"
ECR_IMAGE_URI="${aws_ecr_repository.financial_fraud_training.repository_url}:1.0.1"
NGC_API_KEY="${local.ngc_api_key}"
WRAPPER_S3_PATH="s3://$INPUT_BUCKET/code/training/wrapper.sh"
CONFIG_GEN_S3_PATH="s3://$INPUT_BUCKET/code/training/generate_config.py"
PREPROCESS_DATA_PATH="s3://$OUTPUT_BUCKET/processed/ieee-fraud-detection/"
OUTPUT_DATA_PATH="s3://$OUTPUT_BUCKET/output/ieee-fraud-detection/"

# Generate unique job name with timestamp (matching notebook format)
TIMESTAMP=$(date +%d-%b-%Y-%H-%M-%S)
JOB_NAME="fraud-detection-training-$TIMESTAMP"

echo "=================================================="
echo "SageMaker Training Job Configuration"
echo "=================================================="
echo "Job Name: $JOB_NAME"
echo "Region: $REGION"
echo "Role ARN: $ROLE_ARN"
echo "ECR Image: $ECR_IMAGE_URI"
echo "Input Data: $PREPROCESS_DATA_PATH"
echo "Output: $OUTPUT_DATA_PATH"
echo "Wrapper Script: $WRAPPER_S3_PATH"
echo "=================================================="

# Check if preprocessed data exists
echo "🔍 Checking if preprocessed data exists..."
REQUIRED_PREPROCESS_FILES=(
  "gnn/train_gnn/edges/node_to_node.csv"
  "gnn/train_gnn/nodes/node.csv"
  "gnn/train_gnn/nodes/node_label.csv"
  "gnn/train_gnn/nodes/offset_range_of_training_node.json"
)

ALL_EXISTS=true
for file in "$${REQUIRED_PREPROCESS_FILES[@]}"; do
  if ! aws s3 ls "$PREPROCESS_DATA_PATH$file" >/dev/null 2>&1; then
    echo "  ❌ Missing: $file"
    ALL_EXISTS=false
  else
    echo "  ✅ Found: $file"
  fi
done

if [ "$ALL_EXISTS" = false ]; then
  echo "=================================================="
  echo "❌ Preprocessed data not found!"
  echo "Please ensure preprocessing job has completed successfully."
  echo "=================================================="
  exit 1
fi

# Check if training output already exists
echo "🔍 Checking if training output already exists..."
if aws s3 ls "$OUTPUT_DATA_PATH" --recursive | grep -q "model.tar.gz"; then
  echo "=================================================="
  echo "⚠️  Training output already exists in S3!"
  echo "Skipping training job."
  echo "=================================================="
  exit 0
fi

# Wait for IAM role to propagate
echo "⏳ Waiting 10 seconds for IAM role to propagate..."
sleep 10

# Generate training configuration
echo "📝 Generating training configuration..."
CONFIG_FILE="${path.module}/scripts/training/config.json"
python "${path.module}/scripts/training/generate_config.py" "$CONFIG_FILE"
echo "✅ Configuration generated at $CONFIG_FILE"

# Upload config to S3
echo "📤 Uploading configuration to S3..."
aws s3 cp "$CONFIG_FILE" "s3://$OUTPUT_BUCKET/processed/ieee-fraud-detection/config/config.json"

# Create SageMaker training job request
echo "🚀 Creating SageMaker Training Job..."
REQ_FILE="./training-job-$JOB_NAME.json"

cat > "$REQ_FILE" <<JSON
{
  "TrainingJobName": "$JOB_NAME",
  "RoleArn": "$ROLE_ARN",
  "AlgorithmSpecification": {
    "TrainingImage": "$ECR_IMAGE_URI",
    "TrainingInputMode": "File",
    "ContainerEntrypoint": [
      "bash",
      "/opt/ml/input/data/scripts/wrapper.sh"
    ]
  },
  "InputDataConfig": [
    {
      "ChannelName": "gnn",
      "DataSource": {
        "S3DataSource": {
          "S3Uri": "$${PREPROCESS_DATA_PATH}gnn/train_gnn/",
          "S3DataType": "S3Prefix",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    },
    {
      "ChannelName": "config",
      "DataSource": {
        "S3DataSource": {
          "S3Uri": "s3://$OUTPUT_BUCKET/processed/ieee-fraud-detection/config/",
          "S3DataType": "S3Prefix",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    },
    {
      "ChannelName": "scripts",
      "DataSource": {
        "S3DataSource": {
          "S3Uri": "$WRAPPER_S3_PATH",
          "S3DataType": "S3Prefix",
          "S3DataDistributionType": "FullyReplicated"
        }
      }
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "$OUTPUT_DATA_PATH"
  },
  "ResourceConfig": {
    "InstanceType": "ml.g5.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 30
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 86400
  },
  "Environment": {
    "NIM_DISABLE_MODEL_DOWNLOAD": "true",
    "NGC_API_KEY": "$NGC_API_KEY",
    "PYTHONUNBUFFERED": "1"
  },
  "Tags": [
    {
      "Key": "Project",
      "Value": "aws-nvidia-dgl-gnn-xgboost"
    },
    {
      "Key": "Environment",
      "Value": "dev"
    },
    {
      "Key": "ManagedBy",
      "Value": "opentofu"
    }
  ]
}
JSON

# Create the training job
echo "📝 Creating SageMaker Training Job: $JOB_NAME"
CREATE_OUTPUT=$(aws sagemaker create-training-job --region "$REGION" --cli-input-json file://"$REQ_FILE" 2>&1)
CREATE_EXIT_CODE=$?

if [ $CREATE_EXIT_CODE -ne 0 ]; then
  echo "❌ Failed to create training job!"
  echo "$CREATE_OUTPUT"
  rm -f "$REQ_FILE"
  exit 1
fi

echo "✅ Training job creation request submitted"
echo ""

# Wait for job to appear (max 3 minutes)
echo "⏳ Waiting for job to be registered in SageMaker (max 3 minutes)..."
WAIT_START=$(date +%s)
MAX_WAIT=180
JOB_FOUND=false

while true; do
  ELAPSED=$(($(date +%s) - WAIT_START))
  if [ $ELAPSED -gt $MAX_WAIT ]; then
    echo "❌ Timeout: Job did not appear in SageMaker within 3 minutes"
    echo "Job name: $JOB_NAME"
    rm -f "$REQ_FILE"
    exit 1
  fi
  
  STATUS=$(aws sagemaker describe-training-job --region "$REGION" --training-job-name "$JOB_NAME" --query 'TrainingJobStatus' --output text 2>/dev/null || echo "NOT_FOUND")
  
  if [ "$STATUS" != "NOT_FOUND" ]; then
    echo "✅ Job found! Current status: $STATUS"
    JOB_FOUND=true
    break
  fi
  
  echo "  ⏳ Waiting... ($ELAPSED/$MAX_WAIT seconds)"
  sleep 5
done

if [ "$JOB_FOUND" = false ]; then
  echo "❌ Job was not found after creation"
  rm -f "$REQ_FILE"
  exit 1
fi

echo ""
echo "=================================================="
echo "📊 Monitoring job execution (this may take 12-15 minutes)..."
echo "=================================================="

# Monitor job status until completion
LAST_STATUS=""
while true; do
  STATUS=$(aws sagemaker describe-training-job --region "$REGION" --training-job-name "$JOB_NAME" --query 'TrainingJobStatus' --output text 2>/dev/null || echo "ERROR")
  
  if [ "$STATUS" = "ERROR" ]; then
    echo "❌ Error querying job status"
    rm -f "$REQ_FILE"
    exit 1
  fi
  
  # Only print status if it changed
  if [ "$STATUS" != "$LAST_STATUS" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Status: $STATUS"
    LAST_STATUS="$STATUS"
  fi
  
  if [ "$STATUS" = "Completed" ]; then
    echo "✅ Training job completed successfully!"
    break
  elif [ "$STATUS" = "Failed" ]; then
    REASON=$(aws sagemaker describe-training-job --region "$REGION" --training-job-name "$JOB_NAME" --query 'FailureReason' --output text)
    echo "❌ Training job failed: $REASON"
    rm -f "$REQ_FILE"
    exit 1
  elif [ "$STATUS" = "Stopped" ]; then
    echo "⚠️  Training job was stopped"
    rm -f "$REQ_FILE"
    exit 1
  fi
  
  sleep 30
done

echo "=================================================="
echo "✅ Training job completed!"
echo "=================================================="

# Verify output structure
echo "🔍 Verifying training output in S3..."
EXPECTED_FILES=(
  "model.tar.gz"
)

VERIFICATION_FAILED=false
for file in "$${EXPECTED_FILES[@]}"; do
  FULL_PATH="$OUTPUT_DATA_PATH$JOB_NAME/output/$file"
  if ! aws s3 ls "$FULL_PATH" >/dev/null 2>&1; then
    echo "  ❌ Missing expected file: $file"
    VERIFICATION_FAILED=true
  else
    SIZE=$(aws s3 ls "$FULL_PATH" | awk '{print $3}')
    SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $SIZE / 1000000}")
    echo "  ✅ Verified: $file (size: $SIZE bytes, $SIZE_MB MB)"
  fi
done

# List model output directory
echo ""
echo "📂 Training output directory contents:"
aws s3 ls "$OUTPUT_DATA_PATH$JOB_NAME/" --recursive | head -20

if [ "$VERIFICATION_FAILED" = true ]; then
  echo "=================================================="
  echo "❌ Output verification failed!"
  echo "Some expected files are missing."
  echo "=================================================="
  rm -f "$REQ_FILE"
  exit 1
fi

echo "=================================================="
echo "✅ All training outputs verified successfully!"
echo "=================================================="
echo "Output location: \${OUTPUT_DATA_PATH}\${JOB_NAME}/"
echo ""
echo "Generated model artifacts:"
echo "  - model.tar.gz (257 MB) - Contains:"
echo "    - model_repository/ (ONNX models for Triton)"
echo "    - python_backend_model_repository/ (PyTorch backend models)"
echo "    - training_snapshot/ (comprehensive training metadata)"

# Cleanup
rm -f "$REQ_FILE"
EOT
  }
}

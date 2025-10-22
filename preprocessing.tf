# OpenTofu configuration to create SageMaker Preprocessing Job
# Uses local-exec to launch and monitor the job

# IAM Role for SageMaker Processing
resource "aws_iam_role" "sagemaker_processing" {
  name = "${local.project_name}-sagemaker-processing-role"

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

# Attach SageMaker full access policy
resource "aws_iam_role_policy_attachment" "sagemaker_processing_full_access" {
  role       = aws_iam_role.sagemaker_processing.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Attach S3 full access policy for data access
resource "aws_iam_role_policy_attachment" "sagemaker_processing_s3_access" {
  role       = aws_iam_role.sagemaker_processing.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Upload preprocessing script to S3
resource "aws_s3_object" "preprocessing_script" {
  bucket = aws_s3_bucket.training_input.id
  key    = "code/preprocessing/preprocess.py"
  source = "${path.module}/scripts/preprocessing/preprocess.py"

  depends_on = [
    aws_s3_bucket.training_input
  ]

  tags = merge(local.common_tags, {
    Name = "Preprocessing Script"
  })
}

# Launch and monitor SageMaker preprocessing job using local-exec
resource "null_resource" "run_preprocessing_job" {
  depends_on = [
    aws_iam_role.sagemaker_processing,
    aws_iam_role_policy_attachment.sagemaker_processing_full_access,
    aws_iam_role_policy_attachment.sagemaker_processing_s3_access,
    aws_s3_object.preprocessing_script,
    aws_s3_object.train_transaction_data,
    aws_s3_object.train_identity_data
  ]

  triggers = {
    script_hash      = filesha256("${path.module}/scripts/preprocessing/preprocess.py")
    transaction_hash = try(filesha256("${path.module}/data/train_transaction.csv"), "no-transaction-file")
    identity_hash    = try(filesha256("${path.module}/data/train_identity.csv"), "no-identity-file")
    role_arn         = aws_iam_role.sagemaker_processing.arn
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-lc"]
    command = <<EOT
set -euo pipefail
set +H

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${data.aws_region.current.name}
ROLE_ARN="${aws_iam_role.sagemaker_processing.arn}"
INPUT_BUCKET="${aws_s3_bucket.training_input.bucket}"
OUTPUT_BUCKET="${aws_s3_bucket.training_output.bucket}"
SCRIPT_S3_PATH="s3://$INPUT_BUCKET/code/preprocessing/preprocess.py"
INPUT_DATA_PATH="s3://$INPUT_BUCKET/raw-data/"
OUTPUT_DATA_PATH="s3://$OUTPUT_BUCKET/processed/ieee-fraud-detection/"

# Generate unique job name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="fraud-detection-preprocessing-$TIMESTAMP"

echo "=================================================="
echo "SageMaker Preprocessing Job Configuration"
echo "=================================================="
echo "Job Name: $JOB_NAME"
echo "Region: $REGION"
echo "Role ARN: $ROLE_ARN"
echo "Input: $INPUT_DATA_PATH"
echo "Output: $OUTPUT_DATA_PATH"
echo "Script: $SCRIPT_S3_PATH"
echo "=================================================="

# Check if preprocessed data already exists in S3
echo "🔍 Checking if preprocessed data already exists in S3..."
REQUIRED_FILES=(
  "gnn/train_gnn/edges/node_to_node.csv"
  "gnn/train_gnn/nodes/node.csv"
  "gnn/train_gnn/nodes/node_label.csv"
  "gnn/train_gnn/nodes/offset_range_of_training_node.json"
  "gnn/test/edges/node_to_node.csv"
  "gnn/test/nodes/node.csv"
  "gnn/test/nodes/node_label.csv"
  "xgb/training.csv"
  "xgb/test.csv"
  "xgb/feature_info.json"
)

ALL_EXISTS=true
for file in "$${REQUIRED_FILES[@]}"; do
  if ! aws s3 ls "$OUTPUT_DATA_PATH$file" >/dev/null 2>&1; then
    echo "  ❌ Missing: $file"
    ALL_EXISTS=false
  else
    echo "  ✅ Found: $file"
  fi
done

if [ "$ALL_EXISTS" = true ]; then
  echo "=================================================="
  echo "✅ All preprocessed data already exists in S3!"
  echo "Skipping preprocessing job."
  echo "=================================================="
  exit 0
fi

echo "📦 Preprocessed data not found or incomplete. Creating preprocessing job..."

# Wait for IAM role to propagate (AWS eventual consistency)
echo "⏳ Waiting 10 seconds for IAM role to propagate..."
sleep 10

# Create Processing Job request JSON
echo "🚀 Launching SageMaker preprocessing job via AWS CLI..."
REQ_FILE="./$JOB_NAME.json"
IMAGE_URI="683313688378.dkr.ecr.$REGION.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
cat > "$REQ_FILE" <<JSON
{
  "ProcessingJobName": "$JOB_NAME",
  "RoleArn": "$ROLE_ARN",
  "ProcessingResources": {
    "ClusterConfig": {
      "InstanceCount": 1,
      "InstanceType": "ml.m5.4xlarge",
      "VolumeSizeInGB": 30
    }
  },
  "AppSpecification": {
    "ImageUri": "$IMAGE_URI",
    "ContainerEntrypoint": [
      "python3",
      "/opt/ml/processing/input/code/preprocess.py"
    ],
    "ContainerArguments": [
      "--train-transaction", "/opt/ml/processing/input/train_transaction.csv",
      "--train-identity", "/opt/ml/processing/input/train_identity.csv",
      "--output-path", "/opt/ml/processing/output"
    ]
  },
  "ProcessingInputs": [
    {
      "InputName": "input-data",
      "S3Input": {
        "S3Uri": "$INPUT_DATA_PATH",
        "LocalPath": "/opt/ml/processing/input",
        "S3DataType": "S3Prefix",
        "S3InputMode": "File",
        "S3DataDistributionType": "FullyReplicated",
        "S3CompressionType": "None"
      }
    },
    {
      "InputName": "code",
      "S3Input": {
        "S3Uri": "$SCRIPT_S3_PATH",
        "LocalPath": "/opt/ml/processing/input/code",
        "S3DataType": "S3Prefix",
        "S3InputMode": "File",
        "S3DataDistributionType": "FullyReplicated",
        "S3CompressionType": "None"
      }
    }
  ],
  "ProcessingOutputConfig": {
    "Outputs": [
      {
        "OutputName": "processed-data",
        "S3Output": {
          "S3Uri": "$OUTPUT_DATA_PATH",
          "LocalPath": "/opt/ml/processing/output",
          "S3UploadMode": "EndOfJob"
        }
      }
    ]
  },
  "StoppingCondition": { "MaxRuntimeInSeconds": 7200 },
  "Tags": [
    {"Key": "Project", "Value": "aws-nvidia-dgl-gnn-xgboost"},
    {"Key": "Environment", "Value": "dev"},
    {"Key": "ManagedBy", "Value": "opentofu"}
  ]
}
JSON

# Create the processing job
echo "📝 Creating SageMaker Processing Job: $JOB_NAME"
CREATE_OUTPUT=$(aws sagemaker create-processing-job --region "$REGION" --cli-input-json file://"$REQ_FILE" 2>&1)
CREATE_EXIT_CODE=$?

if [ $CREATE_EXIT_CODE -ne 0 ]; then
  echo "❌ Failed to create processing job!"
  echo "$CREATE_OUTPUT"
  exit 1
fi

echo "✅ Processing job creation request submitted"
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
    exit 1
  fi
  
  STATUS=$(aws sagemaker describe-processing-job --region "$REGION" --processing-job-name "$JOB_NAME" --query 'ProcessingJobStatus' --output text 2>/dev/null || echo "NOT_FOUND")
  
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
  exit 1
fi

echo ""
echo "=================================================="
echo "📊 Monitoring job execution (this may take 10-20 minutes)..."
echo "=================================================="

# Monitor job status until completion
LAST_STATUS=""
while true; do
  STATUS=$(aws sagemaker describe-processing-job --region "$REGION" --processing-job-name "$JOB_NAME" --query 'ProcessingJobStatus' --output text 2>/dev/null || echo "ERROR")
  
  if [ "$STATUS" = "ERROR" ]; then
    echo "❌ Error querying job status"
    exit 1
  fi
  
  # Only print status if it changed
  if [ "$STATUS" != "$LAST_STATUS" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Status: $STATUS"
    LAST_STATUS="$STATUS"
  fi
  
  if [ "$STATUS" = "Completed" ]; then
    echo "✅ Preprocessing job completed successfully!"
    break
  elif [ "$STATUS" = "Failed" ]; then
    REASON=$(aws sagemaker describe-processing-job --region "$REGION" --processing-job-name "$JOB_NAME" --query 'FailureReason' --output text)
    echo "❌ Preprocessing job failed: $REASON"
    exit 1
  elif [ "$STATUS" = "Stopped" ]; then
    echo "⚠️  Preprocessing job was stopped"
    exit 1
  fi
  
  sleep 30
done

echo "=================================================="
echo "✅ Preprocessing job completed!"
echo "=================================================="

# Verify output structure
echo "🔍 Verifying preprocessed data structure in S3..."
VERIFICATION_FAILED=false
for file in "$${REQUIRED_FILES[@]}"; do
  if ! aws s3 ls "$OUTPUT_DATA_PATH$file" >/dev/null 2>&1; then
    echo "  ❌ Missing expected file: $file"
    VERIFICATION_FAILED=true
  else
    SIZE=$(aws s3 ls "$OUTPUT_DATA_PATH$file" | awk '{print $3}')
    echo "  ✅ Verified: $file (size: $SIZE bytes)"
  fi
done

if [ "$VERIFICATION_FAILED" = true ]; then
  echo "=================================================="
  echo "❌ Output verification failed!"
  echo "Some expected files are missing."
  echo "=================================================="
  exit 1
fi

echo "=================================================="
echo "✅ All preprocessing outputs verified successfully!"
echo "=================================================="
echo "Output location: $OUTPUT_DATA_PATH"
echo ""
echo "Generated structure:"
echo "  - gnn/train_gnn/ (edges + nodes with features, labels, offsets)"
echo "  - gnn/test/ (edges + nodes)"
echo "  - xgb/ (training.csv, test.csv, feature_info.json)"

# Cleanup
rm -f "$REQ_FILE"
EOT
  }
}


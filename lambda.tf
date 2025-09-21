# IAM Role for Lambda function
resource "aws_iam_role" "sagemaker_job_lambda" {
  name = "graph-neural-network-demo-sagemaker-job-lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
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

# IAM Policy for Lambda to manage SageMaker jobs
resource "aws_iam_role_policy" "sagemaker_job_lambda" {
  name = "graph-neural-network-demo-sagemaker-job-policy"
  role = aws_iam_role.sagemaker_job_lambda.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutions",
          "sagemaker:DescribePipeline"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
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
      }
    ]
  })
}

# Create Lambda function Python file
resource "local_file" "lambda_bedrock_invoke_py" {
  content = file("${path.module}/lambda/bedrock_invoke/lambda_function.py")
  filename = "${path.module}/lambda/bedrock_invoke/lambda_function.py"
}

# Create __init__.py file for the Lambda package
resource "local_file" "lambda_init_py" {
  content = ""
  filename = "${path.module}/lambda/bedrock_invoke/__init__.py"
}

# Create Lambda function package using archive_file
data "archive_file" "lambda_bedrock_invoke_zip" {
  type        = "zip"
  source_dir  = "${path.module}/lambda/bedrock_invoke"
  output_path = "${path.module}/lambda/bedrock_invoke.zip"
  
  depends_on = [
    local_file.lambda_bedrock_invoke_py,
    local_file.lambda_init_py
  ]
}

# Lambda function for deploying SageMaker training jobs
resource "aws_lambda_function" "deploy_sagemaker_job" {
  function_name = "graph-neural-network-demo-deploy-sagemaker-job"
  role          = aws_iam_role.sagemaker_job_lambda.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  
  # Use the archive file instead of hardcoded filename
  filename         = data.archive_file.lambda_bedrock_invoke_zip.output_path
  source_code_hash = data.archive_file.lambda_bedrock_invoke_zip.output_base64sha256
  
  timeout = 300

  environment {
    variables = {
      TRAINING_INPUT_BUCKET  = aws_s3_bucket.training_input.bucket
      TRAINING_OUTPUT_BUCKET = aws_s3_bucket.training_output.bucket
      REGION                 = "us-east-1"
      # Note: You'll need to create a SageMaker execution role and set it here
      LAMBDA_ROLE_ARN        = aws_iam_role.sagemaker_job_lambda.arn
    }
  }

  tags = {
    Project     = "graph-neural-network-demo"
    Environment = "dev"
    ManagedBy   = "opentofu"
  }

  depends_on = [
    aws_iam_role_policy.sagemaker_job_lambda,
    data.archive_file.lambda_bedrock_invoke_zip,
    aws_s3_object.train_identity_data,
    aws_s3_object.train_transaction_data,
    aws_s3_object.training_script
  ]
}

# Lambda invocation using aws_lambda_invocation resource
resource "aws_lambda_invocation" "trigger_training_job" {
  function_name = aws_lambda_function.deploy_sagemaker_job.function_name

  input = jsonencode({
    action    = "deploy_training_job"
    timestamp = timestamp()
  })

  # Use async invocation type to avoid blocking
  lifecycle {
    ignore_changes = [input]
  }

  depends_on = [
    aws_lambda_function.deploy_sagemaker_job,
    aws_s3_object.train_identity_data,
    aws_s3_object.train_transaction_data,
    aws_s3_object.training_script
  ]
}

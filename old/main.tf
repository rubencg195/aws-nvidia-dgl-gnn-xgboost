terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}

# Output values
output "training_input_bucket" {
  description = "S3 bucket for training input data"
  value       = aws_s3_bucket.training_input.bucket
}

output "training_output_bucket" {
  description = "S3 bucket for training output artifacts"
  value       = aws_s3_bucket.training_output.bucket
}

output "lambda_function_name" {
  description = "Name of the SageMaker job deployment Lambda function"
  value       = aws_lambda_function.deploy_sagemaker_job.function_name
}

output "lambda_invocation_result" {
  description = "Result of the Lambda function invocation"
  value       = aws_lambda_invocation.trigger_training_job.result
  sensitive   = true
}

# Pipeline outputs
output "sagemaker_pipeline_name" {
  description = "Name of the SageMaker pipeline"
  value       = aws_sagemaker_pipeline.graph_neural_network.pipeline_name
}

output "sagemaker_pipeline_arn" {
  description = "ARN of the SageMaker pipeline"
  value       = aws_sagemaker_pipeline.graph_neural_network.arn
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "sagemaker_pipeline_role_arn" {
  description = "ARN of the SageMaker pipeline role"
  value       = aws_iam_role.sagemaker_pipeline.arn
}

# S3 Data Upload Outputs
output "training_data_identity_s3_key" {
  description = "S3 key for the uploaded identity training data"
  value       = aws_s3_object.train_identity_data.key
}

output "training_data_transaction_s3_key" {
  description = "S3 key for the uploaded transaction training data"
  value       = aws_s3_object.train_transaction_data.key
}

output "training_script_s3_key" {
  description = "S3 key for the uploaded training script"
  value       = aws_s3_object.training_script.key
}
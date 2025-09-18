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

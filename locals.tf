# Local values for the aws-nvidia-dgl-gnn-xgboost project
# This file contains all constants and configuration values

locals {
  # Project configuration
  project_name = "aws-nvidia-dgl-gnn-xgboost"
  environment  = "dev"
  managed_by   = "opentofu"

  # Network configuration
  vpc_cidr_block    = "10.0.0.0/16"
  subnet_cidr_blocks = ["10.0.1.0/24", "10.0.2.0/24"]
  default_route     = "0.0.0.0/0"

  # SageMaker Studio configuration
  sagemaker_domain_name = "aws-nvidia-dgl-gnn-xgboost-domain"
  sagemaker_user_profile_name = "admin"

  # Instance types
  sagemaker_instance_types = {
    jupyter_server = "system"
    kernel_gateway = "ml.t3.medium"
  }

  # IAM configuration
  sagemaker_studio_role_name = "${local.project_name}-sagemaker-studio"
  sagemaker_admin_policy_name = "${local.project_name}-sagemaker-admin-policy"

  # Resource names and descriptions
  vpc_name        = "${local.project_name}-vpc"
  igw_name        = "${local.project_name}-igw"
  rt_name         = "${local.project_name}-rt"
  sg_name         = "${local.project_name}-sg"
  sg_description  = "Security group for SageMaker Studio"

  # Tags
  common_tags = {
    Environment = local.environment
    ManagedBy   = local.managed_by
    Project     = local.project_name
  }

  ecr_repo_name          = "financial-fraud-training"
  ecr_image_tag          = "1.0.1"
  nvidia_image_full_name = "nvcr.io/nvidia/cugraph/${local.ecr_repo_name}:${local.ecr_image_tag}"

  # S3 paths
  s3_paths = {
    studio_output     = "s3://${aws_s3_bucket.training_output.bucket}/studio-output/"
    studio_workspace  = "s3://${aws_s3_bucket.training_output.bucket}/studio-workspace/"
    admin_output      = "s3://${aws_s3_bucket.training_output.bucket}/studio-output/admin"
    admin_workspace   = "s3://${aws_s3_bucket.training_output.bucket}/studio-workspace/admin"
  }
}

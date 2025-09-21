# SageMaker Studio Domain and Admin User Configuration
# This creates a SageMaker Studio domain and admin user for pipeline monitoring

# Note: VPC configuration requires manual setup based on your AWS environment
# You need to replace the placeholder VPC and subnet IDs with your actual values

# IAM Role for SageMaker Studio
resource "aws_iam_role" "sagemaker_studio" {
  name = "graph-neural-network-demo-sagemaker-studio"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Environment = "dev"
    ManagedBy   = "opentofu"
    Project     = "graph-neural-network-demo"
  }
}

# Attach SageMaker Full Access Policy
resource "aws_iam_role_policy_attachment" "sagemaker_studio_full_access" {
  role       = aws_iam_role.sagemaker_studio.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Attach SageMaker Admin Policy (custom policy for full admin access)
resource "aws_iam_role_policy" "sagemaker_studio_admin" {
  name = "graph-neural-network-demo-sagemaker-admin-policy"
  role = aws_iam_role.sagemaker_studio.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:*",
          "s3:*",
          "iam:*",
          "logs:*",
          "cloudwatch:*",
          "ec2:*",
          "lambda:*"
        ]
        Resource = "*"
      }
    ]
  })
}

# SageMaker Studio Domain
resource "aws_sagemaker_domain" "graph_neural_network" {
  domain_name = "graph-neural-network-demo-domain"
  auth_mode   = "IAM"
  # VPC configuration - REQUIRED for SageMaker Studio
  # Replace these with your actual VPC and subnet IDs
  vpc_id     = "vpc-12345678"  # TODO: Replace with your VPC ID
  subnet_ids = ["subnet-12345678", "subnet-87654321"]  # TODO: Replace with your subnet IDs

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_studio.arn

    security_groups = []

    sharing_settings {
      notebook_output_option = "Allowed"
      s3_kms_key_id         = null
      s3_output_path        = "s3://${aws_s3_bucket.training_output.bucket}/studio-output/"
    }

    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = "system"
        lifecycle_config_arn = null
      }
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
        lifecycle_config_arn = null
      }
    }

    canvas_app_settings {
      time_series_forecasting_settings {
        amazon_forecast_role_arn = aws_iam_role.sagemaker_studio.arn
        status                   = "ENABLED"
      }
      model_register_settings {
        cross_account_model_register_role_arn = null
        status                               = "ENABLED"
      }
      workspace_settings {
        s3_artifact_path = "s3://${aws_s3_bucket.training_output.bucket}/studio-workspace/"
        s3_kms_key_id    = null
      }
    }
  }

  retention_policy {
    home_efs_file_system = "Delete"
  }

  tags = {
    Environment = "dev"
    ManagedBy   = "opentofu"
    Project     = "graph-neural-network-demo"
  }
}

# SageMaker Studio User Profile for Admin
resource "aws_sagemaker_user_profile" "admin" {
  domain_id         = aws_sagemaker_domain.graph_neural_network.id
  user_profile_name = "admin"

  user_settings {
    execution_role = aws_iam_role.sagemaker_studio.arn

    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = "system"
        lifecycle_config_arn = null
      }
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
        lifecycle_config_arn = null
      }
    }

    canvas_app_settings {
      time_series_forecasting_settings {
        amazon_forecast_role_arn = aws_iam_role.sagemaker_studio.arn
        status                   = "ENABLED"
      }
      model_register_settings {
        cross_account_model_register_role_arn = null
        status                               = "ENABLED"
      }
      workspace_settings {
        s3_artifact_path = "s3://${aws_s3_bucket.training_output.bucket}/studio-workspace/admin"
        s3_kms_key_id    = null
      }
    }

    sharing_settings {
      notebook_output_option = "Allowed"
      s3_kms_key_id         = null
      s3_output_path        = "s3://${aws_s3_bucket.training_output.bucket}/studio-output/admin"
    }
  }

  tags = {
    Environment = "dev"
    ManagedBy   = "opentofu"
    Project     = "graph-neural-network-demo"
    User        = "admin"
  }
}

# Outputs for Studio Access
output "sagemaker_studio_domain_id" {
  description = "SageMaker Studio Domain ID"
  value       = aws_sagemaker_domain.graph_neural_network.id
}

output "sagemaker_studio_domain_name" {
  description = "SageMaker Studio Domain Name"
  value       = aws_sagemaker_domain.graph_neural_network.domain_name
}

output "sagemaker_studio_admin_profile_name" {
  description = "SageMaker Studio Admin User Profile Name"
  value       = aws_sagemaker_user_profile.admin.user_profile_name
}

output "sagemaker_studio_admin_role_arn" {
  description = "SageMaker Studio Admin Role ARN"
  value       = aws_iam_role.sagemaker_studio.arn
}

output "sagemaker_studio_admin_instructions" {
  description = "Instructions for accessing SageMaker Studio"
  value = <<EOF
🎯 To access SageMaker Studio:

1. Open AWS Console: https://console.aws.amazon.com/sagemaker
2. Go to "Studio" in the left navigation
3. Click "Open Studio"
4. Select domain: ${aws_sagemaker_domain.graph_neural_network.domain_name}
5. Select user: ${aws_sagemaker_user_profile.admin.user_profile_name}
6. Click "Open Studio"

📊 You can now:
- View SageMaker Pipelines
- Monitor pipeline executions
- Access pipeline logs
- View model artifacts in S3
- Create notebooks for analysis

🔐 The admin user has full SageMaker permissions to manage and monitor all resources.

🚨 IMPORTANT: Before deploying, update VPC configuration in studio.tf:
   - Replace 'vpc-12345678' with your actual VPC ID
   - Replace subnet IDs with your actual subnet IDs
EOF
}

output "sagemaker_studio_status" {
  description = "Current SageMaker Studio deployment status"
  value = <<EOF
📋 SageMaker Studio Configuration Status:

✅ IAM Role: ${aws_iam_role.sagemaker_studio.name}
✅ Admin Policy: Attached
✅ Studio Domain: ${aws_sagemaker_domain.graph_neural_network.domain_name}
✅ User Profile: ${aws_sagemaker_user_profile.admin.user_profile_name}
❌ VPC Config: Needs manual setup (see instructions above)

🔧 Next Steps:
1. Configure VPC settings in studio.tf
2. Run: tofu apply -auto-approve
3. Access Studio via AWS Console
EOF
}

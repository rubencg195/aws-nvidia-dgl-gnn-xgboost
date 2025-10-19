# SageMaker Studio Domain and Admin User Configuration
# This creates a SageMaker Studio domain and admin user for pipeline monitoring

# VPC resources are now defined in vpc.tf

# IAM Role for SageMaker Studio
resource "aws_iam_role" "sagemaker_studio" {
  name = local.sagemaker_studio_role_name
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

  tags = local.common_tags
}

# Attach SageMaker Full Access Policy
resource "aws_iam_role_policy_attachment" "sagemaker_studio_full_access" {
  role       = aws_iam_role.sagemaker_studio.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Attach SageMaker Admin Policy (custom policy for full admin access)
resource "aws_iam_role_policy" "sagemaker_studio_admin" {
  name = local.sagemaker_admin_policy_name
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
  domain_name = local.sagemaker_domain_name
  auth_mode   = "IAM"
  # VPC configuration - using the created VPC and subnets
  vpc_id     = aws_vpc.sagemaker_studio.id
  subnet_ids = aws_subnet.sagemaker_studio[*].id

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_studio.arn

    security_groups = [aws_security_group.sagemaker_studio.id]

    sharing_settings {
      notebook_output_option = "Allowed"
      s3_kms_key_id         = null
      s3_output_path        = local.s3_paths.studio_output
    }

    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = local.sagemaker_instance_types.jupyter_server
        lifecycle_config_arn = null
      }
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = local.sagemaker_instance_types.kernel_gateway
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
        s3_artifact_path = local.s3_paths.studio_workspace
        s3_kms_key_id    = null
      }
    }
  }

  retention_policy {
    home_efs_file_system = "Delete"
  }

  tags = local.common_tags
}

# SageMaker Studio User Profile for Admin
resource "aws_sagemaker_user_profile" "admin" {
  domain_id         = aws_sagemaker_domain.graph_neural_network.id
  user_profile_name = local.sagemaker_user_profile_name

  user_settings {
    execution_role = aws_iam_role.sagemaker_studio.arn

    jupyter_server_app_settings {
      default_resource_spec {
        instance_type = local.sagemaker_instance_types.jupyter_server
        lifecycle_config_arn = null
      }
    }

    kernel_gateway_app_settings {
      default_resource_spec {
        instance_type = local.sagemaker_instance_types.kernel_gateway
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
        s3_artifact_path = local.s3_paths.admin_workspace
        s3_kms_key_id    = null
      }
    }

    sharing_settings {
      notebook_output_option = "Allowed"
      s3_kms_key_id         = null
      s3_output_path        = local.s3_paths.admin_output
    }
  }

  tags = merge(local.common_tags, {
    User = local.sagemaker_user_profile_name
  })
}

# Outputs for Studio Access
output "sagemaker_studio_domain_id" {
  description = "SageMaker Studio Domain ID"
  value       = aws_sagemaker_domain.graph_neural_network.id
}

output "sagemaker_studio_domain_name" {
  description = "SageMaker Studio Domain Name"
  value       = local.sagemaker_domain_name
}

output "sagemaker_studio_admin_profile_name" {
  description = "SageMaker Studio Admin User Profile Name"
  value       = local.sagemaker_user_profile_name
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
4. Select domain: ${local.sagemaker_domain_name}
5. Select user: ${local.sagemaker_user_profile_name}
6. Click "Open Studio"

📊 You can now:
- View SageMaker Pipelines
- Monitor pipeline executions
- Access pipeline logs
- View model artifacts in S3
- Create notebooks for analysis

🔐 The admin user has full SageMaker permissions to manage and monitor all resources.

✅ VPC Configuration: Created dedicated VPC with subnets and security group
EOF
}

# Debug outputs to see VPC information
output "debug_vpc_info" {
  description = "Debug information about VPC configuration"
  value = <<EOF
📋 VPC Configuration Debug Information:

Created VPC ID: ${aws_vpc.sagemaker_studio.id}
Created Subnets: ${join(", ", aws_subnet.sagemaker_studio[*].id)}
Created Security Group: ${aws_security_group.sagemaker_studio.id}
Internet Gateway: ${aws_internet_gateway.sagemaker_studio.id}
Route Table: ${aws_route_table.sagemaker_studio.id}

VPC Configuration Status:
- VPC Created: ✅
- Subnets Created: ✅ (${length(aws_subnet.sagemaker_studio)} subnets)
- Security Group Created: ✅
- Internet Gateway Created: ✅
- Route Table Created: ✅

Network Configuration:
- VPC CIDR: ${local.vpc_cidr_block}
- Subnet 1 CIDR: ${local.subnet_cidr_blocks[0]}
- Subnet 2 CIDR: ${local.subnet_cidr_blocks[1]}
- Internet Access: ✅ (via IGW)
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
✅ VPC Config: Created dedicated VPC with subnets and security group

🔧 Next Steps:
1. Run: tofu apply -auto-approve
2. Access Studio via AWS Console
EOF
}

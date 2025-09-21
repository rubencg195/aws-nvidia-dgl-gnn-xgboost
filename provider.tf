# Configure the AWS Provider
provider "aws" {
  region = local.aws_region

  default_tags {
    tags = local.common_tags
  }
}

# Local variables
locals {
  aws_region   = "us-east-1"
  common_tags = {
    Project     = "graph-neural-network-demo"
    Environment = "dev"
    ManagedBy   = "opentofu"
  }
}

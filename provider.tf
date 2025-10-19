# Configure the AWS Provider
provider "aws" {
  region = local.aws_region

  default_tags {
    tags = local.common_tags
  }
}

# Local variables
locals {
  aws_region = "us-east-1"
}

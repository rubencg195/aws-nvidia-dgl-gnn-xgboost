# VPC Configuration for SageMaker Studio
# This file contains all VPC-related resources

# Get availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Create a VPC for SageMaker Studio
resource "aws_vpc" "sagemaker_studio" {
  cidr_block = local.vpc_cidr_block

  tags = merge(local.common_tags, {
    Name = local.vpc_name
  })
}

# Create subnets for SageMaker Studio
resource "aws_subnet" "sagemaker_studio" {
  count = 2

  vpc_id     = aws_vpc.sagemaker_studio.id
  cidr_block = local.subnet_cidr_blocks[count.index]

  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-subnet-${count.index + 1}"
  })
}

# Create internet gateway for the VPC
resource "aws_internet_gateway" "sagemaker_studio" {
  vpc_id = aws_vpc.sagemaker_studio.id

  tags = merge(local.common_tags, {
    Name = local.igw_name
  })
}

# Create route table
resource "aws_route_table" "sagemaker_studio" {
  vpc_id = aws_vpc.sagemaker_studio.id

  route {
    cidr_block = local.default_route
    gateway_id = aws_internet_gateway.sagemaker_studio.id
  }

  tags = merge(local.common_tags, {
    Name = local.rt_name
  })
}

# Associate route table with subnets
resource "aws_route_table_association" "sagemaker_studio" {
  count = 2

  subnet_id      = aws_subnet.sagemaker_studio[count.index].id
  route_table_id = aws_route_table.sagemaker_studio.id
}

# Create security group for SageMaker Studio
resource "aws_security_group" "sagemaker_studio" {
  name        = local.sg_name
  description = local.sg_description
  vpc_id      = aws_vpc.sagemaker_studio.id

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [local.default_route]
  }

  tags = merge(local.common_tags, {
    Name = local.sg_name
  })
}

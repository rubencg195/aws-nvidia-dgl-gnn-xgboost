#!/usr/bin/env python3
"""
Lambda function to deploy SageMaker training jobs for financial fraud detection
"""

import json
import boto3
import os
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
sagemaker = boto3.client('sagemaker', region_name=os.environ['REGION'])
s3 = boto3.client('s3', region_name=os.environ['REGION'])


def lambda_handler(event, context):
    """
    Main Lambda handler to deploy SageMaker training job
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get environment variables
        training_input_bucket = os.environ['TRAINING_INPUT_BUCKET']
        training_output_bucket = os.environ['TRAINING_OUTPUT_BUCKET']
        region = os.environ['REGION']
        
        # Generate unique job name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f"fraud-detection-xgboost-{timestamp}"
        
        # SageMaker training job configuration
        training_job_config = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
                "TrainingInputMode": "File"
            },
            "RoleArn": os.environ['LAMBDA_ROLE_ARN'],  # This should be the SageMaker execution role
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{training_input_bucket}/train/",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "text/csv"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{training_output_bucket}/models/"
            },
            "ResourceConfig": {
                "InstanceType": "ml.m5.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 86400  # 24 hours
            },
            "HyperParameters": {
                "max_depth": "6",
                "eta": "0.3",
                "objective": "binary:logistic",
                "num_round": "100",
                "eval_metric": "auc"
            },
            "Environment": {
                "SAGEMAKER_PROGRAM": "train.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": f"s3://{training_input_bucket}/code/train.py"
            }
        }
        
        logger.info(f"Creating training job: {job_name}")
        
        # Create the training job
        response = sagemaker.create_training_job(**training_job_config)
        
        logger.info(f"Training job created successfully: {response['TrainingJobArn']}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Training job started successfully",
                "job_name": job_name,
                "job_arn": response['TrainingJobArn']
            })
        }
        
    except Exception as e:
        logger.error(f"Error creating training job: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Failed to create training job",
                "error": str(e)
            })
        }

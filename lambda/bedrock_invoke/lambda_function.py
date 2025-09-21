#!/usr/bin/env python3
"""
Lambda function to trigger SageMaker pipeline execution for financial fraud detection
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
    Main Lambda handler to start SageMaker pipeline execution
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")

        # Get environment variables
        region = os.environ['REGION']
        pipeline_name = "graph-neural-network-demo-pipeline"

        # Generate unique execution name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        execution_name = f"pipeline-execution-{timestamp}"

        logger.info(f"Starting pipeline execution: {execution_name}")
        logger.info("Pipeline is configured with static parameters - no runtime parameters needed")

        # Start pipeline execution (no parameters since pipeline uses static configuration)
        response = sagemaker.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionName=execution_name
        )

        logger.info(f"Pipeline execution started successfully: {response['PipelineExecutionArn']}")

        # Get pipeline execution details
        execution_details = sagemaker.describe_pipeline_execution(
            PipelineExecutionArn=response['PipelineExecutionArn']
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Pipeline execution started successfully",
                "pipeline_name": pipeline_name,
                "execution_name": execution_name,
                "execution_arn": response['PipelineExecutionArn'],
                "execution_status": execution_details.get('PipelineExecutionStatus', 'Unknown'),
                "creation_time": str(execution_details.get('CreationTime', ''))
            })
        }

    except sagemaker.exceptions.ResourceNotFound as e:
        logger.error(f"Pipeline not found: {str(e)}")
        return {
            "statusCode": 404,
            "body": json.dumps({
                "message": "Pipeline not found",
                "error": "Please ensure the SageMaker pipeline has been created",
                "pipeline_name": "graph-neural-network-demo-pipeline"
            })
        }

    except Exception as e:
        logger.error(f"Error starting pipeline execution: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Failed to start pipeline execution",
                "error": str(e)
            })
        }


def get_pipeline_status(pipeline_execution_arn):
    """
    Get the status of a pipeline execution
    """
    try:
        response = sagemaker.describe_pipeline_execution(
            PipelineExecutionArn=pipeline_execution_arn
        )
        return response.get('PipelineExecutionStatus', 'Unknown')
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        return "Error"


def list_pipeline_executions(pipeline_name, max_results=10):
    """
    List recent pipeline executions
    """
    try:
        response = sagemaker.list_pipeline_executions(
            PipelineName=pipeline_name,
            MaxResults=max_results,
            SortBy='CreationTime',
            SortOrder='Descending'
        )
        return response.get('PipelineExecutionSummaries', [])
    except Exception as e:
        logger.error(f"Error listing pipeline executions: {str(e)}")
        return []

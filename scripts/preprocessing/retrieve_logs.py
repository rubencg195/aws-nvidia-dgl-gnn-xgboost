#!/usr/bin/env python3
"""
Retrieve and print CloudWatch logs for SageMaker Preprocessing Jobs
"""

import boto3
import sys
import json
from datetime import datetime

def get_preprocessing_job_logs(job_name=None, region="us-east-1"):
    """
    Retrieve and print CloudWatch logs for a preprocessing job
    
    Args:
        job_name: The SageMaker preprocessing job name (if None, retrieves latest)
        region: AWS region
    """
    
    # Initialize clients
    sagemaker = boto3.client('sagemaker', region_name=region)
    logs = boto3.client('logs', region_name=region)
    
    try:
        # If no job name provided, find the latest preprocessing job
        if not job_name:
            response = sagemaker.list_processing_jobs(
                SortOrder='Descending',
                SortBy='CreationTime',
                MaxResults=1
            )
            
            if not response['ProcessingJobSummaries']:
                print("❌ No preprocessing jobs found")
                return False
            
            job_name = response['ProcessingJobSummaries'][0]['ProcessingJobName']
        
        print(f"📊 Retrieving logs for job: {job_name}")
        print("=" * 80)
        
        # Get job details
        job_response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
        status = job_response['ProcessingJobStatus']
        print(f"Job Status: {status}")
        print(f"Creation Time: {job_response['CreationTime']}")
        if job_response.get('ProcessingEndTime'):
            print(f"End Time: {job_response['ProcessingEndTime']}")
        print("=" * 80)
        
        # CloudWatch log group for SageMaker processing jobs
        log_group_name = f"/aws/sagemaker/ProcessingJobs/{job_name}"
        
        print(f"\n📝 CloudWatch Log Group: {log_group_name}\n")
        
        try:
            # Get log streams
            streams_response = logs.describe_log_streams(
                logGroupName=log_group_name,
                orderBy='LastEventTime',
                descending=True
            )
            
            if not streams_response['logStreams']:
                print("⚠️  No log streams found")
                return False
            
            # Print logs from each stream
            for stream in streams_response['logStreams']:
                stream_name = stream['logStreamName']
                print(f"\n📄 Stream: {stream_name}")
                print("-" * 80)
                
                # Get log events
                events_response = logs.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=stream_name,
                    startFromHead=True
                )
                
                for event in events_response['events']:
                    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                    message = event['message'].strip()
                    print(f"[{timestamp}] {message}")
            
            print("\n" + "=" * 80)
            print("✅ Log retrieval complete")
            return True
            
        except logs.exceptions.ResourceNotFoundException:
            print(f"⚠️  Log group not found: {log_group_name}")
            print("This may indicate the job hasn't started yet or failed to create logs.")
            return False
            
    except Exception as e:
        print(f"❌ Error retrieving logs: {str(e)}")
        return False


if __name__ == "__main__":
    job_name = sys.argv[1] if len(sys.argv) > 1 else None
    region = sys.argv[2] if len(sys.argv) > 2 else "us-east-1"
    
    success = get_preprocessing_job_logs(job_name, region)
    sys.exit(0 if success else 1)

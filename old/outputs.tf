# Additional outputs for pipeline monitoring and debugging

# Pipeline monitoring information
output "pipeline_monitoring_info" {
  description = "Information for monitoring pipeline execution"
  value = <<EOF
To monitor your pipeline execution:

1. Run the monitoring script (recommended):
   ./scripts/pipeline-monitoring.sh

2. Check pipeline executions:
   aws sagemaker list-pipeline-executions --pipeline-name graph-neural-network-demo-pipeline

3. Get detailed execution status:
   aws sagemaker describe-pipeline-execution --pipeline-execution-arn <PIPELINE_ARN>

4. Check individual job status:
   aws sagemaker describe-training-job --training-job-name <JOB_NAME>

5. View logs in CloudWatch:
   - Log group: /aws/sagemaker/TrainingJobs
   - Log group: /aws/sagemaker/ProcessingJobs
   - Log group: /aws/lambda/graph-neural-network-demo-deploy-sagemaker-job

6. Check S3 for model artifacts:
   - Input bucket: ${aws_s3_bucket.training_input.bucket}
   - Output bucket: ${aws_s3_bucket.training_output.bucket}

The monitoring script will automatically:
- Detect the latest pipeline execution ARN
- Monitor status every 2 minutes for 5 iterations
- Search CloudWatch logs for error keywords on failure
- Provide targeted error information using regex pattern matching
EOF
}

# Quick commands for monitoring
output "quick_monitoring_commands" {
  description = "Quick commands for monitoring pipeline execution"
  value = <<EOF
# Run the automated monitoring script (recommended)
./scripts/pipeline-monitoring.sh

# List recent pipeline executions
aws sagemaker list-pipeline-executions --pipeline-name graph-neural-network-demo-pipeline --max-results 5

# Get the latest pipeline execution ARN
aws sagemaker list-pipeline-executions --pipeline-name graph-neural-network-demo-pipeline --max-results 1 --sort-by CreationTime --sort-order Descending --query 'PipelineExecutionSummaries[0].PipelineExecutionArn' --output text

# Describe the latest pipeline execution
aws sagemaker describe-pipeline-execution --pipeline-execution-arn $(aws sagemaker list-pipeline-executions --pipeline-name graph-neural-network-demo-pipeline --max-results 1 --sort-by CreationTime --sort-order Descending --query 'PipelineExecutionSummaries[0].PipelineExecutionArn' --output text)

# Check training job status (replace JOB_NAME with actual job name)
aws sagemaker describe-training-job --training-job-name <JOB_NAME>

# Check processing job status (replace JOB_NAME with actual job name)
aws sagemaker describe-processing-job --processing-job-name <JOB_NAME>

# List CloudWatch log groups
aws logs describe-log-groups | grep sagemaker

# Search for errors in specific log group
aws logs tail --log-group-name <log-group-name> --max-items 1000 | grep -i 'error\|exception\|warn\|fail'

# Use the monitoring script for comprehensive error detection
./scripts/pipeline-monitoring.sh
EOF
}

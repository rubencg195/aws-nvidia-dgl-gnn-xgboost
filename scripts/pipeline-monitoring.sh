#!/bin/bash

# Pipeline Monitoring Script
# Monitors SageMaker pipeline execution and searches CloudWatch logs for errors on failures
# Usage: ./scripts/pipeline-monitoring.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PIPELINE_NAME="graph-neural-network-demo-pipeline"
MAX_ITERATIONS=5
WAIT_SECONDS=120
LAMBDA_RESPONSE_FILE="lambda_response.json"
ERROR_KEYWORDS="error|exception|warn|warning|fail|failed|timeout|accessdenied|validationerror|outofmemory"

echo -e "${BLUE}🚀 Starting Pipeline Monitoring${NC}"
echo "========================================"

# Function to get pipeline ARN
get_pipeline_arn() {
    echo -e "${YELLOW}🔍 Obtaining Pipeline ARN...${NC}"

    # First, try to get ARN from lambda response
    if [ -f "$LAMBDA_RESPONSE_FILE" ]; then
        echo "Found lambda response file"
        PIPELINE_ARN=$(grep -o '"execution_arn":"[^"]*"' "$LAMBDA_RESPONSE_FILE" | cut -d'"' -f4)

        if [ -n "$PIPELINE_ARN" ] && [ "$PIPELINE_ARN" != "null" ]; then
            echo -e "${GREEN}✅ Pipeline ARN from lambda response: ${PIPELINE_ARN}${NC}"
            return 0
        fi
    fi

    # Alternative: Get the latest pipeline execution ARN
    echo "Lambda response not available, getting latest pipeline execution..."
    PIPELINE_ARN=$(aws sagemaker list-pipeline-executions \
        --pipeline-name "$PIPELINE_NAME" \
        --max-results 1 \
        --sort-by CreationTime \
        --sort-order Descending \
        --query 'PipelineExecutionSummaries[0].PipelineExecutionArn' \
        --output text 2>/dev/null)

    if [ -z "$PIPELINE_ARN" ] || [ "$PIPELINE_ARN" = "None" ]; then
        echo -e "${RED}❌ Could not find pipeline ARN${NC}"
        echo "Please check manually or run the lambda function first."
        exit 1
    fi

    echo -e "${GREEN}✅ Latest Pipeline ARN: ${PIPELINE_ARN}${NC}"
}

# Function to search CloudWatch logs for error-related keywords
search_cloudwatch_logs() {
    local log_group="$1"
    local keywords="${2:-error|exception|warn|warning|fail|failed|timeout|accessdenied}"

    echo -e "${YELLOW}🔍 Searching for errors in ${log_group}...${NC}"

    if aws logs describe-log-streams --log-group-name "$log_group" --max-items 1 >/dev/null 2>&1; then
        # Get log events and filter for error keywords
        local temp_file=$(mktemp)
        aws logs tail --log-group-name "$log_group" --max-items 1000 --output json 2>/dev/null > "$temp_file"

        if [ -s "$temp_file" ]; then
        # Extract and filter log messages
        jq -r '.events[]?.message // empty' "$temp_file" 2>/dev/null | grep -i "$ERROR_KEYWORDS" | head -20
            if [ $? -eq 0 ]; then
                echo ""
            else
                echo "No error-related messages found in recent logs"
            fi
        else
            echo "No logs available in $log_group"
        fi

        rm -f "$temp_file"
    else
        echo "Log group $log_group does not exist or has no streams"
    fi
}

# Function to get processing job logs
get_processing_job_logs() {
    echo -e "${YELLOW}🔍 Searching for errors in Processing Job logs...${NC}"

    # Get all log groups related to SageMaker
    LOG_GROUPS=$(aws logs describe-log-groups --query 'logGroups[*].logGroupName' --output text 2>/dev/null | tr '\t' '\n' | grep -i sagemaker)

    for log_group in $LOG_GROUPS; do
        if [[ "$log_group" == *"ProcessingJobs"* ]]; then
            echo -e "${BLUE}📄 Log Group: $log_group${NC}"
            search_cloudwatch_logs "$log_group" "$ERROR_KEYWORDS"
            echo "---"
        fi
    done
}

# Function to get training job logs
get_training_job_logs() {
    echo -e "${YELLOW}🔍 Searching for errors in Training Job logs...${NC}"

    # Get all log groups related to SageMaker
    LOG_GROUPS=$(aws logs describe-log-groups --query 'logGroups[*].logGroupName' --output text 2>/dev/null | tr '\t' '\n' | grep -i sagemaker)

    for log_group in $LOG_GROUPS; do
        if [[ "$log_group" == *"TrainingJobs"* ]]; then
            echo -e "${BLUE}📄 Log Group: $log_group${NC}"
            search_cloudwatch_logs "$log_group" "$ERROR_KEYWORDS"
            echo "---"
        fi
    done
}

# Function to get lambda function logs
get_lambda_logs() {
    echo -e "${YELLOW}🔍 Searching for errors in Lambda logs...${NC}"

    # Get Lambda log group
    LAMBDA_LOG_GROUP="/aws/lambda/graph-neural-network-demo-deploy-sagemaker-job"

    echo -e "${BLUE}📄 Log Group: $LAMBDA_LOG_GROUP${NC}"
    search_cloudwatch_logs "$LAMBDA_LOG_GROUP" "$ERROR_KEYWORDS"
    echo "---"
}

# Function to get pipeline step details
get_step_details() {
    echo -e "${YELLOW}📊 Getting Step Execution Details...${NC}"

    # Get step executions
    aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$PIPELINE_ARN" \
        --query 'StepExecutions[*].[StepName, StepStatus, FailureReason, StartTime, EndTime]' \
        --output table 2>/dev/null || echo "No step execution details available"
}

# Main monitoring function
monitor_pipeline() {
    local PIPELINE_ARN="$1"
    local FINAL_STATUS=""

    echo -e "${BLUE}🎯 Starting Pipeline Monitoring${NC}"
    echo "========================================"

    # Monitor pipeline status for specified iterations
    for i in $(seq 1 $MAX_ITERATIONS); do
        echo -e "${YELLOW}=== Pipeline Status Check $i/$MAX_ITERATIONS ===${NC}"
        echo "Timestamp: $(date)"

        # Get pipeline execution details
        STATUS=$(aws sagemaker describe-pipeline-execution \
            --pipeline-execution-arn "$PIPELINE_ARN" \
            --query 'PipelineExecutionStatus' \
            --output text 2>/dev/null)

        if [ $? -eq 0 ] && [ -n "$STATUS" ] && [ "$STATUS" != "None" ]; then
            echo -e "${BLUE}📊 Pipeline Status: $STATUS${NC}"

            # Show step statuses if available
            echo -e "${BLUE}📋 Step Executions:${NC}"
            get_step_details

        else
            echo -e "${RED}❌ Error getting pipeline status${NC}"
            STATUS="Error"
        fi

        # If pipeline failed, get detailed logs
        if [ "$STATUS" = "Failed" ]; then
            echo -e "${RED}🚨 Pipeline execution failed! Searching for errors in logs...${NC}"

            # Get processing job logs
            get_processing_job_logs

            # Get training job logs
            get_training_job_logs

            # Get lambda function logs
            get_lambda_logs

            FINAL_STATUS="$STATUS"
            break
        fi

        # If pipeline succeeded, exit early
        if [ "$STATUS" = "Succeeded" ]; then
            echo -e "${GREEN}✅ Pipeline execution succeeded!${NC}"
            FINAL_STATUS="$STATUS"
            break
        fi

        # Wait before next check (only if not the last iteration)
        if [ $i -lt $MAX_ITERATIONS ]; then
            echo -e "${YELLOW}⏳ Waiting $WAIT_SECONDS seconds before next check...${NC}"
            sleep $WAIT_SECONDS
        fi
    done

    echo "========================================"
    echo -e "${BLUE}=== Pipeline Monitoring Complete ===${NC}"
    echo -e "${BLUE}📊 Final Status: ${FINAL_STATUS:-Unknown}${NC}"
    echo -e "${BLUE}🔗 Pipeline ARN: $PIPELINE_ARN${NC}"

    # Provide helpful next steps
    echo ""
    echo -e "${YELLOW}🔧 For more details, run:${NC}"
    echo "aws sagemaker describe-pipeline-execution --pipeline-execution-arn $PIPELINE_ARN"
    echo ""
    echo -e "${YELLOW}📋 To search CloudWatch logs for errors manually:${NC}"
    echo "aws logs describe-log-groups | grep sagemaker"
    echo "aws logs tail --log-group-name <log-group-name> --max-items 1000 | grep -i 'error\|exception\|warn\|fail'"

    return 0
}

# Main execution
main() {
    echo -e "${GREEN}🖥️  Pipeline Monitoring Script${NC}"
    echo "============================================"
    echo "Monitoring pipeline: $PIPELINE_NAME"
    echo "Max iterations: $MAX_ITERATIONS"
    echo "Check interval: $WAIT_SECONDS seconds"
    echo "Error keywords: error|exception|warn|warning|fail|failed|timeout|accessdenied|validationerror|outofmemory"
    echo ""

    # Get pipeline ARN
    get_pipeline_arn

    # Start monitoring
    monitor_pipeline "$PIPELINE_ARN"
}

# Run main function
main "$@"

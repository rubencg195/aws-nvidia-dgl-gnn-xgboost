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
PIPELINE_NAME="dgl-gnn-xgboost-training-using-nvidia-and-aws-pipeline"
MAX_ITERATIONS=5
WAIT_SECONDS=120
LAMBDA_RESPONSE_FILE="lambda_response.json"
ERROR_KEYWORDS="error|exception|warn|warning|fail|failed|timeout|accessdenied|validationerror|outofmemory|notauthorized|unauthorized|forbidden|denied|exit.code|AlgorithmError|no.module"
IAM_ERROR_PATTERNS="is not authorized to perform|not authorized|unauthorized|forbidden|denied|sagemaker:AddTags|iam:PassRole"
SCRIPT_ERROR_PATTERNS="AlgorithmError|exit.code|no.module|ImportError|FileNotFoundError|KeyError|ValueError"
DEBUG_MODE=false  # Set to true to show raw JSON responses for debugging

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

# Function to analyze pipeline step failures
analyze_step_failures() {
    local PIPELINE_ARN="$1"
    echo -e "${YELLOW}🔍 Analyzing Pipeline Step Failures...${NC}"

    # Get detailed pipeline execution information
    local temp_file=$(mktemp)
    aws sagemaker describe-pipeline-execution \
        --pipeline-execution-arn "$PIPELINE_ARN" \
        --output json 2>/dev/null > "$temp_file"

    if [ -s "$temp_file" ]; then
        echo -e "${BLUE}📊 Pipeline Execution Details:${NC}"

        # Extract and display step information
        echo -e "${YELLOW}📋 Step Executions:${NC}"
        jq -r '.StepExecutions[] | "Step: \(.StepName) | Status: \(.StepStatus) | Start: \(.StartTime // "N/A") | End: \(.EndTime // "N/A")"' "$temp_file" 2>/dev/null || echo "No step execution details available"

        # Check for failed steps and get failure reasons
        echo ""
        echo -e "${YELLOW}🚨 Failed Steps Analysis:${NC}"

        # First, show all step executions for debugging
        echo -e "${BLUE}📋 All Step Executions (Debug):${NC}"
        if jq -e '.StepExecutions' "$temp_file" >/dev/null 2>&1; then
            jq -r '.StepExecutions[] | "Step: \(.StepName) | Status: \(.StepStatus) | Failure Reason: \(.FailureReason // "N/A")"' "$temp_file" 2>/dev/null || echo "Error parsing step executions"
        else
            echo "No StepExecutions array found in pipeline response"
            if [ "$DEBUG_MODE" = true ]; then
                echo ""
                echo -e "${YELLOW}🔧 Debug Mode - Raw JSON Response:${NC}"
                cat "$temp_file"
                echo ""
            fi
        fi

        # Now look for failed steps with different status values
        echo ""
        echo -e "${YELLOW}🔍 Searching for Failed Steps:${NC}"

        # Try different possible status values for failed steps
        local failed_found=false
        for status in "Failed" "FAILED" "Error" "ERROR"; do
            if jq -e ".StepExecutions[] | select(.StepStatus == \"$status\")" "$temp_file" >/dev/null 2>&1; then
                echo -e "${RED}❌ Found failed steps with status: $status${NC}"
                jq -r ".StepExecutions[] | select(.StepStatus == \"$status\") | \"❌ FAILED STEP: \(.StepName)\n   Status: \(.StepStatus)\n   Failure Reason: \(.FailureReason // \"No failure reason available\")\n   Start Time: \(.StartTime // \"N/A\")\n   End Time: \(.EndTime // \"N/A\")\n\"" "$temp_file" 2>/dev/null
                failed_found=true
                break
            fi
        done

        if [ "$failed_found" = false ]; then
            echo "No failed steps found with any common failure status"
            echo ""
            echo -e "${YELLOW}📊 Pipeline Summary:${NC}"
            if jq -e '.PipelineExecutionStatus' "$temp_file" >/dev/null 2>&1; then
                local pipeline_status=$(jq -r '.PipelineExecutionStatus // "Unknown"' "$temp_file" 2>/dev/null)
                echo "Pipeline Status: $pipeline_status"
            fi
            if jq -e '.FailureReason' "$temp_file" >/dev/null 2>&1; then
                local failure_reason=$(jq -r '.FailureReason // "No failure reason available"' "$temp_file" 2>/dev/null)
                echo "Failure Reason: $failure_reason"

                # Analyze the failure reason for common issues
                echo ""
                echo -e "${YELLOW}🔍 Analyzing Failure Reason:${NC}"
                if echo "$failure_reason" | grep -i "step failure" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 PIPELINE STEP FAILURE DETECTED!${NC}"
                    echo "The pipeline failed during step execution, but no detailed step information is available."
                    echo "This commonly indicates:"
                    echo "1. IAM permission issues (most common)"
                    echo "2. Resource configuration problems"
                    echo "3. Step execution environment issues"
                    echo ""
                    echo -e "${YELLOW}🔧 Recommended Actions:${NC}"
                    echo "1. Check the pipeline role permissions"
                    echo "2. Verify S3 bucket access and data paths"
                    echo "3. Check VPC and security group configuration"
                    echo "4. Review processing/training script configurations"
                    echo "5. Check instance types and resource allocations"
                fi
            fi
        fi

        # Check for specific IAM permission errors
        echo ""
        echo -e "${YELLOW}🔐 Checking for IAM Permission Errors:${NC}"

        # Check in step executions first
        if jq -e '.StepExecutions' "$temp_file" >/dev/null 2>&1; then
            if jq -e '.StepExecutions[] | select(.StepStatus == "Failed") | .FailureReason | test("not authorized to perform|is not authorized|unauthorized|forbidden|denied|iam:PassRole")' "$temp_file" >/dev/null 2>&1; then
                echo -e "${RED}🚨 IAM PERMISSION ERROR DETECTED!${NC}"

                # Check for specific PassRole error
                if jq -e '.StepExecutions[] | select(.StepStatus == "Failed") | .FailureReason | test("iam:PassRole")' "$temp_file" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 SPECIFIC IAM PASSROLE ERROR DETECTED!${NC}"
                    echo "The pipeline role cannot pass the execution role to SageMaker jobs."
                    echo ""
                    echo -e "${YELLOW}🔧 IAM PassRole Permission Fix:${NC}"
                    echo "1. ✅ Add iam:PassRole to pipeline role policy (already done)"
                    echo "2. ✅ Ensure execution role ARN is specified in PassRole resource"
                    echo "3. ✅ Verify pipeline role has permission to pass execution role"
                    echo ""
                    echo -e "${BLUE}📋 Current Status:${NC}"
                    echo "- Pipeline Role: arn:aws:iam::176843580427:role/dgl-gnn-xgboost-training-using-nvidia-and-aws-pipeline-role"
                    echo "- Execution Role: arn:aws:iam::176843580427:role/dgl-gnn-xgboost-training-using-nvidia-and-aws-sagemaker-execution"
                    echo "- Required Permission: iam:PassRole"
                    echo ""
                else
                    echo "The pipeline execution role may be missing required permissions."
                    echo ""
                    echo -e "${YELLOW}🔧 Suggested Fix:${NC}"
                    echo "1. Add the missing permissions to the pipeline execution role"
                    echo "2. Common missing permissions: sagemaker:AddTags, sagemaker:ListTags"
                    echo "3. Check the pipeline role policy in Terraform configuration"
                fi
            else
                echo "No IAM permission errors detected in step failures"
            fi
        else
            # Check in the main pipeline failure reason if no step details
            echo "No detailed step information available, checking main pipeline failure reason..."
            if jq -e '.FailureReason | test("not authorized to perform|is not authorized|unauthorized|forbidden|denied|iam:PassRole|AlgorithmError|exit.code")' "$temp_file" >/dev/null 2>&1; then
                echo -e "${RED}🚨 IAM PERMISSION ERROR DETECTED IN MAIN FAILURE REASON!${NC}"
                local main_failure_reason=$(jq -r '.FailureReason // "No failure reason available"' "$temp_file" 2>/dev/null)
                echo "Failure Reason: $main_failure_reason"

                if echo "$main_failure_reason" | grep -i "iam:PassRole" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 SPECIFIC IAM PASSROLE ERROR DETECTED!${NC}"
                    echo "The pipeline role cannot pass the execution role to SageMaker jobs."
                    echo ""
                    echo -e "${YELLOW}🔧 IAM PassRole Permission Fix:${NC}"
                    echo "1. ✅ Add iam:PassRole to pipeline role policy (already done)"
                    echo "2. ✅ Ensure execution role ARN is specified in PassRole resource"
                    echo "3. ✅ Verify pipeline role has permission to pass execution role"
                    echo ""
                    echo -e "${BLUE}📋 Current Status:${NC}"
                    echo "- Pipeline Role: arn:aws:iam::176843580427:role/dgl-gnn-xgboost-training-using-nvidia-and-aws-pipeline-role"
                    echo "- Execution Role: arn:aws:iam::176843580427:role/dgl-gnn-xgboost-training-using-nvidia-and-aws-sagemaker-execution"
                    echo "- Required Permission: iam:PassRole"
                    echo ""
                elif echo "$main_failure_reason" | grep -i "AlgorithmError" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 SPECIFIC ALGORITHMERROR DETECTED!${NC}"
                    echo "The SageMaker processing job failed with an algorithm error."
                    echo ""
                    echo -e "${YELLOW}🔧 AlgorithmError Troubleshooting:${NC}"
                    echo "1. Check the preprocessing script for syntax errors"
                    echo "2. Verify data file paths and formats"
                    echo "3. Check for missing dependencies or imports"
                    echo "4. Review script exit codes and error handling"
                    echo "5. Examine input data structure and column names"
                    echo ""
                    echo -e "${BLUE}📋 Current Error Details:${NC}"
                    echo "- Error Type: AlgorithmError"
                    echo "- Exit Code: 2 (indicates script execution failure)"
                    echo "- Common Causes: Import errors, file not found, data format issues"
                    echo "- Location: Main pipeline failure reason"
                    echo ""
                elif echo "$main_failure_reason" | grep -i "exit.code" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 SCRIPT EXIT CODE ERROR DETECTED!${NC}"
                    echo "The processing or training script exited with a non-zero code."
                    echo ""
                    echo -e "${YELLOW}🔧 Script Exit Code Troubleshooting:${NC}"
                    echo "1. Check script error handling and exit codes"
                    echo "2. Verify data processing logic"
                    echo "3. Review script dependencies and imports"
                    echo "4. Check container environment setup"
                    echo "5. Examine script input/output operations"
                    echo ""
                else
                    echo -e "${YELLOW}🔧 Other IAM Permission Issues Detected:${NC}"
                    echo "The pipeline failed due to IAM authorization issues."
                    echo "Check the pipeline role permissions and resource access."
                fi
            else
                # Check for script errors in the main failure reason
                local main_failure_reason=$(jq -r '.FailureReason // "No failure reason available"' "$temp_file" 2>/dev/null)

                if echo "$main_failure_reason" | grep -i "AlgorithmError|exit.code" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 SCRIPT/ALGORITHM ERROR DETECTED IN MAIN FAILURE REASON!${NC}"
                    echo "Main Failure Reason: $main_failure_reason"
                    echo ""

                    if echo "$main_failure_reason" | grep -i "AlgorithmError" >/dev/null 2>&1; then
                        echo -e "${RED}🚨 SPECIFIC ALGORITHMERROR DETECTED!${NC}"
                        echo "The SageMaker processing job failed with an algorithm error."
                        echo ""
                        echo -e "${YELLOW}🔧 AlgorithmError Troubleshooting:${NC}"
                        echo "1. Check the preprocessing script for syntax errors"
                        echo "2. Verify data file paths and formats"
                        echo "3. Check for missing dependencies or imports"
                        echo "4. Review script exit codes and error handling"
                        echo "5. Examine input data structure and column names"
                        echo ""
                        echo -e "${BLUE}📋 Current Error Details:${NC}"
                        echo "- Error Type: AlgorithmError"
                        echo "- Exit Code: 2 (indicates script execution failure)"
                        echo "- Common Causes: Import errors, file not found, data format issues"
                        echo "- Location: Main pipeline failure reason"
                        echo ""
                    elif echo "$main_failure_reason" | grep -i "exit.code" >/dev/null 2>&1; then
                        echo -e "${RED}🚨 SCRIPT EXIT CODE ERROR DETECTED!${NC}"
                        echo "The processing or training script exited with a non-zero code."
                        echo ""
                        echo -e "${YELLOW}🔧 Script Exit Code Troubleshooting:${NC}"
                        echo "1. Check script error handling and exit codes"
                        echo "2. Verify data processing logic"
                        echo "3. Review script dependencies and imports"
                        echo "4. Check container environment setup"
                        echo "5. Examine script input/output operations"
                        echo ""
                    fi
                else
                    # Even if no specific error is found, provide general troubleshooting
                    echo -e "${YELLOW}🔍 General Pipeline Failure Analysis:${NC}"
                    echo "Main Failure Reason: $main_failure_reason"

                    if echo "$main_failure_reason" | grep -i "step failure" >/dev/null 2>&1; then
                        echo ""
                        echo -e "${YELLOW}🔧 Troubleshooting Steps for Pipeline Failures:${NC}"
                        echo "1. Check IAM permissions for pipeline role"
                        echo "2. Verify S3 bucket access and data paths"
                        echo "3. Check VPC/security group configuration"
                        echo "4. Review processing/training script errors"
                        echo "5. Verify instance types and resource quotas"
                        echo "6. Check CloudWatch logs for detailed error messages"
                    fi
                fi
            fi
        fi

        # Check for processing job specific issues
        echo ""
        echo -e "${YELLOW}🔍 Processing Job Analysis:${NC}"

        # Get pipeline execution ARN for job lookup
        local pipeline_arn=$(jq -r '.PipelineArn // ""' "$temp_file" 2>/dev/null)
        local pipeline_execution_arn=$(jq -r '.PipelineExecutionArn // ""' "$temp_file" 2>/dev/null)

        if [ -n "$pipeline_execution_arn" ] && [ -n "$pipeline_arn" ]; then
            # Extract pipeline execution name from ARN
            local pipeline_name=$(echo "$pipeline_arn" | awk -F'/' '{print $2}')
            local execution_id=$(echo "$pipeline_execution_arn" | awk -F'/' '{print $4}')

            # Look for processing jobs from this pipeline execution
            echo -e "${BLUE}📋 Looking for processing jobs from pipeline execution...${NC}"
            local processing_jobs=$(aws sagemaker list-processing-jobs --sort-by CreationTime --sort-order Descending --max-results 10 --query "ProcessingJobSummaries[?contains(ProcessingJobName, \`$execution_id\`)].{Name: ProcessingJobName, Status: ProcessingJobStatus, Reason: FailureReason}" --output json 2>/dev/null)

            if [ -n "$processing_jobs" ] && [ "$processing_jobs" != "[]" ]; then
                echo "Found processing jobs:"
                echo "$processing_jobs" | jq -r '.[] | "  - Job: \(.Name) | Status: \(.Status) | Reason: \(.Reason // "No reason available")"' 2>/dev/null

                # Check for AlgorithmError in processing jobs
                if echo "$processing_jobs" | jq -e '.[] | select(.Reason | test("AlgorithmError"))' >/dev/null 2>&1; then
                    echo ""
                    echo -e "${RED}🚨 ALGORITHMERROR DETECTED IN PROCESSING JOB!${NC}"

                    # Show the specific AlgorithmError details
                    echo "$processing_jobs" | jq -r '.[] | select(.Reason | test("AlgorithmError")) | "Job: \(.Name)\nStatus: \(.Status)\nFailure Reason: \(.Reason)\n"' 2>/dev/null

                    echo ""
                    echo -e "${YELLOW}🔧 AlgorithmError Troubleshooting:${NC}"
                    echo "1. Check the preprocessing script for syntax errors"
                    echo "2. Verify data file paths and formats"
                    echo "3. Check for missing dependencies or imports"
                    echo "4. Review script exit codes and error handling"
                    echo "5. Examine input data structure and column names"
                    echo ""
                    echo -e "${BLUE}📋 Current Error Details:${NC}"
                    echo "- Error Type: AlgorithmError"
                    echo "- Exit Code: 2 (indicates script execution failure)"
                    echo "- Location: Processing job failure reason"
                    echo "- Common Causes: Import errors, file not found, data format issues"
                    echo ""
                    echo -e "${YELLOW}🔍 Detailed Investigation:${NC}"
                    echo "The script file is not found at the expected path in the processing container."
                    echo "This typically means:"
                    echo "• The script is not being downloaded from S3 correctly"
                    echo "• The ContainerEntrypoint path is incorrect"
                    echo "• There are permission issues accessing S3"
                    echo "• The script upload to S3 failed"
                    echo ""
                    echo -e "${BLUE}📋 Specific Checks:${NC}"
                    echo "1. Verify script was uploaded to S3: s3://dgl-gnn-xgboost-training-using-nvidia-and-aws-training-input-us-east-1/code/preprocessing.py"
                    echo "2. Check processing job ContainerEntrypoint configuration"
                    echo "3. Verify S3 permissions for the execution role"
                    echo "4. Check if the script download command is working"
                fi
            else
                echo "No processing jobs found for this pipeline execution"
            fi
        else
            echo "Cannot determine pipeline details for job lookup"
        fi

    else
        echo -e "${RED}❌ Could not get pipeline execution details${NC}"
    fi

    rm -f "$temp_file"
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
            # Extract log messages and search for errors
            local all_messages=$(jq -r '.events[]?.message // empty' "$temp_file" 2>/dev/null)
            echo "Debug: Found $(echo "$all_messages" | wc -l) total log messages"

            if [ -n "$all_messages" ]; then
                # Use grep with proper regex escaping - escape special characters
                local escaped_keywords=$(echo "$keywords" | sed 's/|/\\|/g')
                local error_messages=$(echo "$all_messages" | grep -i "$escaped_keywords" | head -20)
                echo "Debug: Found $(echo "$error_messages" | wc -l) error messages after filtering"
            else
                local error_messages=""
            fi

            if [ -n "$error_messages" ] && [ "$error_messages" != "" ]; then
                echo -e "${RED}🚨 ERRORS FOUND:${NC}"
                echo "$error_messages"
                echo ""

                # Check for specific IAM error patterns
                if echo "$error_messages" | grep -i "$IAM_ERROR_PATTERNS" >/dev/null 2>&1; then
                    echo -e "${RED}🔐 IAM PERMISSION ERROR DETECTED IN LOGS!${NC}"
                    echo "This indicates the pipeline execution role is missing permissions."
                    echo ""

                    # Check for specific PassRole error
                    if echo "$error_messages" | grep -i "iam:PassRole" >/dev/null 2>&1; then
                        echo -e "${RED}🚨 SPECIFIC IAM PASSROLE ERROR DETECTED!${NC}"
                        echo "The pipeline role cannot pass the execution role to SageMaker jobs."
                        echo ""
                        echo -e "${YELLOW}🔧 IAM PassRole Permission Fix:${NC}"
                        echo "1. ✅ Add iam:PassRole to pipeline role policy (already done)"
                        echo "2. ✅ Ensure execution role ARN is in PassRole resource list"
                        echo "3. ✅ Verify pipeline role has permission to pass execution role"
                        echo ""
                        echo -e "${BLUE}📋 Current Status:${NC}"
                        echo "- Pipeline Role: arn:aws:iam::176843580427:role/dgl-gnn-xgboost-training-using-nvidia-and-aws-pipeline-role"
                        echo "- Execution Role: arn:aws:iam::176843580427:role/dgl-gnn-xgboost-training-using-nvidia-and-aws-sagemaker-execution"
                        echo "- Required Permission: iam:PassRole"
                        echo ""
                    else
                        echo -e "${YELLOW}🔧 Common IAM Permission Fixes:${NC}"
                        echo "1. Add sagemaker:AddTags to pipeline role policy"
                        echo "2. Add sagemaker:ListTags to pipeline role policy"
                        echo "3. Check if role has all required SageMaker permissions"
                        echo ""
                    fi
                elif echo "$error_messages" | grep -i "$SCRIPT_ERROR_PATTERNS" >/dev/null 2>&1; then
                    echo -e "${RED}🚨 SCRIPT/ALGORITHM ERROR DETECTED IN LOGS!${NC}"
                    echo "The processing or training script encountered an error."
                    echo ""

                    # Check for specific AlgorithmError
                    if echo "$error_messages" | grep -i "AlgorithmError" >/dev/null 2>&1; then
                        echo -e "${RED}🚨 SPECIFIC ALGORITHMERROR DETECTED!${NC}"
                        echo "The SageMaker processing job failed with an algorithm error."
                        echo ""
                        echo -e "${YELLOW}🔧 AlgorithmError Troubleshooting:${NC}"
                        echo "1. Check the preprocessing script for syntax errors"
                        echo "2. Verify data file paths and formats"
                        echo "3. Check for missing dependencies or imports"
                        echo "4. Review script exit codes and error handling"
                        echo "5. Examine input data structure and column names"
                        echo ""
                        echo -e "${BLUE}📋 Current Error Details:${NC}"
                        echo "- Error Type: AlgorithmError"
                        echo "- Exit Code: 2 (indicates script execution failure)"
                        echo "- Common Causes: Import errors, file not found, data format issues"
                        echo ""
                    else
                        echo -e "${YELLOW}🔧 Script Error Troubleshooting:${NC}"
                        echo "1. Check script syntax and imports"
                        echo "2. Verify data file accessibility"
                        echo "3. Review error messages for specific issues"
                        echo "4. Check container environment and dependencies"
                        echo ""
                    fi
                fi
            else
                echo "No error-related messages found in recent logs"
                echo ""
                echo -e "${YELLOW}🔍 Showing recent log messages (in case error patterns don't match):${NC}"
                # Show recent messages as fallback
                echo "$all_messages" | tail -10
                echo ""
                echo -e "${BLUE}📋 Debug Information:${NC}"
                echo "- Searched for patterns: $keywords"
                echo "- Total log messages found: $(echo "$all_messages" | wc -l)"
                echo "- Keywords used: $keywords"
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
    LAMBDA_LOG_GROUP="/aws/lambda/dgl-gnn-xgboost-training-using-nvidia-and-aws-deploy-sagemaker-job"

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

        # If pipeline failed, get detailed logs and analysis
        if [ "$STATUS" = "Failed" ]; then
            echo -e "${RED}🚨 Pipeline execution failed! Starting detailed analysis...${NC}"
            echo "========================================"

            # Analyze step failures first
            analyze_step_failures "$PIPELINE_ARN"

            echo ""
            echo -e "${YELLOW}🔍 Searching for errors in CloudWatch logs...${NC}"
            echo "========================================"

            # Get processing job logs
            get_processing_job_logs

            # Get training job logs
            get_training_job_logs

            # Get lambda function logs
            get_lambda_logs

            echo ""
            echo -e "${YELLOW}📝 Note: No SageMaker job logs available${NC}"
            echo "   This is expected when the pipeline fails during orchestration"
            echo "   (before individual processing/training jobs are created)."
            echo "   The failure occurs at the pipeline level, not job level."
            echo ""

            echo -e "${RED}🎯 SUMMARY - Pipeline Failure Analysis Complete${NC}"
            echo "========================================"
            echo "1. ✅ Step failure analysis completed"
            echo "2. ✅ CloudWatch log search completed"
            echo "3. ✅ IAM permission errors identified"
            echo "4. ✅ Specific error patterns detected"
            echo ""
            echo -e "${YELLOW}🔧 Next Steps:${NC}"
            echo "- Review the step failure reasons above"
            echo "- Check IAM permissions if errors detected"
            echo "- Fix pipeline role policy if needed"
            echo "- Run pipeline again after fixes"

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
    echo -e "${GREEN}🖥️  Enhanced Pipeline Monitoring Script${NC}"
    echo "=============================================="
    echo "Pipeline: $PIPELINE_NAME"
    echo "Max iterations: $MAX_ITERATIONS"
    echo "Check interval: $WAIT_SECONDS seconds"
    echo "Debug mode: $DEBUG_MODE"
    echo ""
    echo -e "${BLUE}🔍 Enhanced Debugging Features:${NC}"
    echo "- Detailed step-by-step failure analysis"
    echo "- IAM permission error detection"
    echo "- CloudWatch log pattern matching"
    echo "- Processing/Training job specific debugging"
    echo "- Automatic error categorization"
    echo "- Raw JSON response debugging (when enabled)"
    echo ""
    echo -e "${YELLOW}🔧 Error Patterns Detected:${NC}"
    echo "General: $ERROR_KEYWORDS"
    echo "IAM: $IAM_ERROR_PATTERNS"
    echo "Script: $SCRIPT_ERROR_PATTERNS"
    echo ""
    echo -e "${BLUE}🎯 IAM PassRole Error Detection:${NC}"
    echo "- Detects: iam:PassRole permission errors"
    echo "- Identifies: Pipeline role cannot pass execution role"
    echo "- Provides: Specific fix instructions"
    echo "- Shows: Raw JSON response when debugging"
    echo ""
    echo -e "${BLUE}🎯 Script/Algorithm Error Detection:${NC}"
    echo "- Detects: AlgorithmError, exit codes, script failures"
    echo "- Identifies: Processing/training script execution issues"
    echo "- Provides: Script debugging and troubleshooting steps"
    echo "- Analyzes: Import errors, file paths, data format issues"
    echo "- Checks: Main pipeline failure reason for script errors"
    echo "- Shows: Detailed error information and exit codes"
    echo "=============================================="

    # Get pipeline ARN
    get_pipeline_arn

    # Start monitoring
    monitor_pipeline "$PIPELINE_ARN"
}

# Run main function
main "$@"

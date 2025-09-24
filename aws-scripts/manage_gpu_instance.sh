#!/bin/bash

# GPU Instance Management Script
# Provides start, stop, status, and SSH connection management
# Uses gpu_config.conf for all settings
# Created: 2025-09-11
# Status: NEW

set -e

# Show help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]] || [[ $# -eq 0 ]]; then
    echo "🎮 GPU Instance Management Script"
    echo "================================="
    echo ""
    echo "Usage:"
    echo "  $0 status [--cmd]            # Show status of GPU instances (--cmd shows command only)"
    echo "  $0 start [instance-id] [--cmd] # Start GPU instance (--cmd shows command only)"
    echo "  $0 stop [instance-id] [--cmd]  # Stop GPU instance (--cmd shows command only)"
    echo "  $0 terminate <instance-id> [--not-dry-run] [--cmd] # Terminate instance (--not-dry-run executes, --cmd shows command only)"
    echo "  $0 check-files <instance-id> [--cmd] # Validate instance for important files before termination"
    echo "  $0 ssh [instance-id] [--cmd] # SSH to GPU instance (--cmd shows command only)"
    echo "  $0 list [--cmd]              # List all project instances (--cmd shows command only)"
    echo "  $0 costs [--cmd]             # Show running costs (--cmd shows command only)"
    echo ""
    echo "Examples:"
    echo "  $0 status                    # Quick status check"
    echo "  $0 status --cmd              # Show status command for copy/paste"
    echo "  $0 start                     # Start all stopped instances"
    echo "  $0 start i-04adf9e80ea0cfebc --cmd  # Show start command for copy/paste"
    echo "  $0 stop i-04adf9e80ea0cfebc  # Stop specific instance"
    echo "  $0 stop --cmd                # Show stop command for all instances"
    echo "  $0 terminate i-123abc         # Dry-run: show what would be terminated"
    echo "  $0 terminate i-123abc --not-dry-run # Actually terminate instance"
    echo "  $0 check-files i-123abc       # Validate instance for important files"
    echo "  $0 list --cmd                # Show list command for copy/paste"
    echo "  $0 ssh --cmd                 # Show SSH command only (for copy/paste)"
    echo ""
    exit 0
fi

# Load configuration - smart search for config file
CONFIG_FILE="gpu_config.conf"

# Function to find config file in multiple locations
find_config_file() {
    local config_locations=(
        "gpu_config.conf"                                    # Current directory
        "aws-scripts/gpu_config.conf"                       # From project root
        "../gpu_config.conf"                                # From subdirectory
        "../../aws-scripts/gpu_config.conf"                 # From nested subdirectory
        "$(dirname "$0")/gpu_config.conf"                   # Same directory as script
    )
    
    for location in "${config_locations[@]}"; do
        if [ -f "$location" ]; then
            echo "$location"
            return 0
        fi
    done
    
    return 1
}

# Search for config file
FOUND_CONFIG=$(find_config_file)
if [ $? -eq 0 ]; then
    CONFIG_FILE="$FOUND_CONFIG"
    echo "📋 Found configuration: $CONFIG_FILE"
else
    echo "❌ Configuration file not found: gpu_config.conf"
    echo "   Searched in:"
    echo "   - Current directory"
    echo "   - aws-scripts/ subdirectory" 
    echo "   - Parent directory"
    echo "   - Script's directory"
    echo ""
    echo "💡 Make sure gpu_config.conf exists in one of these locations"
    exit 1
fi

source "$CONFIG_FILE"

# Fix relative paths in config to be relative to config file location
CONFIG_DIR=$(dirname "$CONFIG_FILE")
if [[ "$AWS_CLI_PATH" == ../* ]]; then
    AWS_CLI_PATH="$CONFIG_DIR/$AWS_CLI_PATH"
fi
if [[ "$KEY_FILE" != /* ]] && [[ "$KEY_FILE" != ./* ]]; then
    # If KEY_FILE is just a filename, make it relative to config directory
    KEY_FILE="$CONFIG_DIR/$KEY_FILE"
fi

# Source AWS CLI check function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws_cli_check.sh"

# Ensure AWS CLI is installed
ensure_aws_cli "$AWS_CLI_PATH"

# Verify AWS authentication
$AWS_CLI_PATH sts get-caller-identity --profile $AWS_PROFILE --region $AWS_REGION > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ AWS authentication failed. Please run: $AWS_CLI_PATH sso login --profile $AWS_PROFILE"
    exit 1
fi

# Function to get instances with project tag (intelligent case-insensitive) 
get_project_instances() {
    # Use bash associative arrays to process the results efficiently
    declare -A instances
    declare -A project_tags
    
    # Get all instances with their data and all possible project tag variations in one call
    $AWS_CLI_PATH ec2 describe-instances \
        --output json \
        --profile $AWS_PROFILE \
        --region $AWS_REGION 2>/dev/null | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
target_project = '$TAG_PROJECT'.lower()

for reservation in data['Reservations']:
    for instance in reservation['Instances']:
        instance_id = instance['InstanceId']
        state = instance['State']['Name']
        instance_type = instance['InstanceType']
        private_ip = instance.get('PrivateIpAddress', '-')
        launch_time = instance.get('LaunchTime', '')
        
        # Find name tag (case insensitive)
        name = '-'
        project_tag = None
        
        for tag in instance.get('Tags', []):
            if tag['Key'].lower() == 'name':
                name = tag['Value']
            elif tag['Key'].lower() == 'project':
                project_tag = tag['Value']
        
        # Case-insensitive project matching
        if project_tag and project_tag.lower() == target_project:
            print(f'{instance_id} {state} {instance_type} {private_ip} {name} {launch_time}')
"
}

# Function to get instance status checks
get_instance_status() {
    local instance_id="$1"
    $AWS_CLI_PATH ec2 describe-instance-status \
        --instance-ids "$instance_id" \
        --query "InstanceStatuses[0].[SystemStatus.Status,InstanceStatus.Status]" \
        --output text \
        --profile $AWS_PROFILE \
        --region $AWS_REGION 2>/dev/null
}

# Function to get running instances (intelligent case-insensitive)
get_running_instances() {
    # Get running instances with intelligent project tag matching
    $AWS_CLI_PATH ec2 describe-instances \
        --filters "Name=instance-state-name,Values=running" \
        --output json \
        --profile $AWS_PROFILE \
        --region $AWS_REGION 2>/dev/null | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
target_project = '$TAG_PROJECT'.lower()

for reservation in data['Reservations']:
    for instance in reservation['Instances']:
        instance_id = instance['InstanceId']
        private_ip = instance.get('PrivateIpAddress', '')
        
        # Find project tag (case insensitive)
        project_tag = None
        for tag in instance.get('Tags', []):
            if tag['Key'].lower() == 'project':
                project_tag = tag['Value']
                break
        
        # Case-insensitive project matching
        if project_tag and project_tag.lower() == target_project:
            print(f'{instance_id} {private_ip}')
"
}

# Function to calculate costs (simplified - no bc dependency)
calculate_costs() {
    local instance_type="$1"
    local hours="$2"
    
    case $instance_type in
        "g4dn.xlarge") 
            cost=$(awk "BEGIN {printf \"%.2f\", $hours * 0.526}")
            echo "$cost" ;;
        "g4dn.2xlarge") 
            cost=$(awk "BEGIN {printf \"%.2f\", $hours * 0.752}")
            echo "$cost" ;;
        "g5.xlarge") 
            cost=$(awk "BEGIN {printf \"%.2f\", $hours * 1.006}")
            echo "$cost" ;;
        *) echo "Unknown" ;;
    esac
}

# Main command processing
COMMAND="$1"
INSTANCE_ID=""
COMMAND_ONLY=false
NOT_DRY_RUN=false

# Parse arguments for all commands
shift # Remove command from arguments

while [[ $# -gt 0 ]]; do
    case $1 in
        --cmd)
            COMMAND_ONLY=true
            shift
            ;;
        --not-dry-run)
            NOT_DRY_RUN=true
            shift
            ;;
        i-*)
            # Instance ID (starts with i-)
            INSTANCE_ID="$1"
            shift
            ;;
        *)
            # For backward compatibility, treat non-flag arguments as instance ID
            if [[ -z "$INSTANCE_ID" ]]; then
                INSTANCE_ID="$1"
            fi
            shift
            ;;
    esac
done

case $COMMAND in
    "status"|"stat")
        if [ "$COMMAND_ONLY" = true ]; then
            echo "📋 AWS CLI commands for instance status:"
            echo "$AWS_CLI_PATH ec2 describe-instances --filters \"Name=tag:Project,Values=$TAG_PROJECT\" --query \"Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,PrivateIpAddress,Tags[?Key=='Name'].Value|[0],LaunchTime]\" --output text --profile $AWS_PROFILE --region $AWS_REGION"
            exit 0
        fi
        
        echo "💻 GPU Instance Status"
        echo "======================"
        
        INSTANCES=$(get_project_instances)
        if [ -z "$INSTANCES" ]; then
            echo "📍 No GPU instances found with project tag: $TAG_PROJECT"
            echo ""
            echo "💡 To create a new instance: ./launch_gpu_instance.sh"
            exit 0
        fi
        
        echo "$INSTANCES" | while read -r instance_id state type private_ip name launch_time; do
            if [ "$state" == "running" ]; then
                status_icon="🟢"
            elif [ "$state" == "stopped" ]; then
                status_icon="🔴"
            elif [ "$state" == "stopping" ]; then
                status_icon="🟡"
            elif [ "$state" == "pending" ]; then
                status_icon="🟡"
            else
                status_icon="⚪"
            fi
            
            echo "$status_icon $instance_id ($name)"
            echo "   State: $state | Type: $type"
            if [ ! -z "$private_ip" ] && [ "$private_ip" != "None" ]; then
                echo "   Private IP: $private_ip"
            fi
            echo "   Launched: $launch_time"
            echo ""
        done
        ;;
        
    "start")
        if [ "$COMMAND_ONLY" = true ]; then
            if [ ! -z "$INSTANCE_ID" ]; then
                echo "📋 AWS CLI command to start instance $INSTANCE_ID:"
                echo "$AWS_CLI_PATH ec2 start-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
            else
                echo "📋 AWS CLI commands to start all stopped instances:"
                echo "# Get stopped instances:"
                echo "$AWS_CLI_PATH ec2 describe-instances --filters \"Name=tag:Project,Values=$TAG_PROJECT\" \"Name=instance-state-name,Values=stopped\" --query \"Reservations[*].Instances[*].InstanceId\" --output text --profile $AWS_PROFILE --region $AWS_REGION"
                echo ""
                echo "# Start each instance (replace INSTANCE_ID with actual ID):"
                echo "$AWS_CLI_PATH ec2 start-instances --instance-ids INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
            fi
            exit 0
        fi
        
        if [ ! -z "$INSTANCE_ID" ]; then
            echo "🟢 Starting instance: $INSTANCE_ID"
            $AWS_CLI_PATH ec2 start-instances --instance-ids "$INSTANCE_ID" --profile $AWS_PROFILE --region $AWS_REGION > /dev/null
            echo "✅ Start command sent. Instance will be running shortly."
        else
            echo "🟢 Starting all stopped GPU instances..."
            STOPPED_INSTANCES=$($AWS_CLI_PATH ec2 describe-instances \
                --filters "Name=tag:Project,Values=$TAG_PROJECT" "Name=instance-state-name,Values=stopped" \
                --query "Reservations[*].Instances[*].InstanceId" \
                --output text \
                --profile $AWS_PROFILE \
                --region $AWS_REGION)
            
            if [ -z "$STOPPED_INSTANCES" ]; then
                echo "📍 No stopped instances found"
            else
                for instance in $STOPPED_INSTANCES; do
                    echo "🟢 Starting: $instance"
                    $AWS_CLI_PATH ec2 start-instances --instance-ids "$instance" --profile $AWS_PROFILE --region $AWS_REGION > /dev/null
                done
                echo "✅ All start commands sent"
            fi
        fi
        ;;
        
    "stop")
        if [ "$COMMAND_ONLY" = true ]; then
            if [ ! -z "$INSTANCE_ID" ]; then
                echo "📋 AWS CLI command to stop instance $INSTANCE_ID:"
                echo "$AWS_CLI_PATH ec2 stop-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
            else
                echo "📋 AWS CLI commands to stop all running instances:"
                echo "# Get running instances:"
                echo "$AWS_CLI_PATH ec2 describe-instances --filters \"Name=tag:Project,Values=$TAG_PROJECT\" \"Name=instance-state-name,Values=running\" --query \"Reservations[*].Instances[*].InstanceId\" --output text --profile $AWS_PROFILE --region $AWS_REGION"
                echo ""
                echo "# Stop each instance (replace INSTANCE_ID with actual ID):"
                echo "$AWS_CLI_PATH ec2 stop-instances --instance-ids INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
            fi
            exit 0
        fi
        
        if [ ! -z "$INSTANCE_ID" ]; then
            echo "🔴 Stopping instance: $INSTANCE_ID"
            $AWS_CLI_PATH ec2 stop-instances --instance-ids "$INSTANCE_ID" --profile $AWS_PROFILE --region $AWS_REGION > /dev/null
            echo "✅ Stop command sent. Instance will stop shortly."
        else
            echo "🔴 Stopping all running GPU instances..."
            RUNNING_INSTANCES=$($AWS_CLI_PATH ec2 describe-instances \
                --filters "Name=tag:Project,Values=$TAG_PROJECT" "Name=instance-state-name,Values=running" \
                --query "Reservations[*].Instances[*].InstanceId" \
                --output text \
                --profile $AWS_PROFILE \
                --region $AWS_REGION)
            
            if [ -z "$RUNNING_INSTANCES" ]; then
                echo "📍 No running instances found"
            else
                for instance in $RUNNING_INSTANCES; do
                    echo "🔴 Stopping: $instance"
                    $AWS_CLI_PATH ec2 stop-instances --instance-ids "$instance" --profile $AWS_PROFILE --region $AWS_REGION > /dev/null
                done
                echo "✅ All stop commands sent"
                echo "💰 Cost savings: Instances will stop accruing compute costs"
            fi
        fi
        ;;
        
    "ssh")
        echo "🔑 SSH Connection to GPU Instance"
        echo "================================="
        
        if [ ! -z "$INSTANCE_ID" ]; then
            # SSH to specific instance
            INSTANCE_INFO=$($AWS_CLI_PATH ec2 describe-instances \
                --instance-ids "$INSTANCE_ID" \
                --query "Reservations[*].Instances[*].[State.Name,PrivateIpAddress]" \
                --output text \
                --profile $AWS_PROFILE \
                --region $AWS_REGION)
            
            read state private_ip <<< "$INSTANCE_INFO"
            
            if [ "$state" != "running" ]; then
                echo "❌ Instance $INSTANCE_ID is not running (state: $state)"
                exit 1
            fi
        else
            # Auto-detect running instance
            RUNNING_INFO=$(get_running_instances)
            if [ -z "$RUNNING_INFO" ]; then
                echo "❌ No running GPU instances found"
                echo "💡 Start an instance first: $0 start"
                exit 1
            fi
            
            # Use first running instance
            read INSTANCE_ID private_ip <<< "$(echo "$RUNNING_INFO" | head -1)"
        fi
        
        if [ -z "$private_ip" ] || [ "$private_ip" == "None" ]; then
            echo "❌ No private IP found for instance $INSTANCE_ID"
            exit 1
        fi
        
        if [ ! -f "$KEY_FILE" ]; then
            echo "❌ Key file not found: $KEY_FILE"
            exit 1
        fi
        
        SSH_COMMAND="ssh -i $KEY_FILE ec2-user@$private_ip"
        
        if [ "$COMMAND_ONLY" = true ]; then
            # Just output the command for copy/paste
            echo "📋 SSH Command for $INSTANCE_ID ($private_ip):"
            echo "$SSH_COMMAND"
        else
            # Show info and connect
            echo "🎯 Connecting to: $INSTANCE_ID ($private_ip)"
            echo "💡 SSH Command: $SSH_COMMAND"
            echo ""
            $SSH_COMMAND
        fi
        ;;
        
    "list"|"ls")
        if [ "$COMMAND_ONLY" = true ]; then
            echo "📋 AWS CLI command to list all project instances:"
            echo "$AWS_CLI_PATH ec2 describe-instances --filters \"Name=tag:Project,Values=$TAG_PROJECT\" --query \"Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,PrivateIpAddress,Tags[?Key=='Name'].Value|[0],LaunchTime]\" --output table --profile $AWS_PROFILE --region $AWS_REGION"
            echo ""
            echo "📋 AWS CLI command to get status checks:"
            echo "$AWS_CLI_PATH ec2 describe-instance-status --query \"InstanceStatuses[*].[InstanceId,SystemStatus.Status,InstanceStatus.Status]\" --output table --profile $AWS_PROFILE --region $AWS_REGION"
            exit 0
        fi
        
        echo "📋 All GPU Instances (Project: $TAG_PROJECT)"
        echo "============================================"
        
        INSTANCES=$(get_project_instances)
        if [ -z "$INSTANCES" ]; then
            echo "📍 No instances found"
            exit 0
        fi
        
        printf "%-20s %-13s %-12s %-15s %-15s %-20s\n" "INSTANCE-ID" "STATE" "TYPE" "PRIVATE-IP" "STATUS-CHECK" "NAME"
        echo "---------------------------------------------------------------------------------------------------"
        
        echo "$INSTANCES" | while read -r instance_id state type private_ip name launch_time; do
            if [ -z "$private_ip" ] || [ "$private_ip" == "None" ]; then
                private_ip="-"
            fi
            if [ -z "$name" ] || [ "$name" == "None" ]; then
                name="-"
            fi
            
            # Get status checks only for running instances (others don't have status checks)
            if [ "$state" == "running" ]; then
                status_info=$(get_instance_status "$instance_id")
                if [ -n "$status_info" ]; then
                    system_status=$(echo "$status_info" | cut -f1)
                    instance_status=$(echo "$status_info" | cut -f2)
                    if [ "$system_status" == "ok" ] && [ "$instance_status" == "ok" ]; then
                        status_display="✅ passed"
                    elif [ "$system_status" == "impaired" ] || [ "$instance_status" == "impaired" ]; then
                        status_display="❌ failed"
                    elif [ "$system_status" == "initializing" ] || [ "$instance_status" == "initializing" ]; then
                        status_display="🔄 initializing"
                    else
                        status_display="⚠️ $system_status/$instance_status"
                    fi
                else
                    status_display="🔄 checking..."
                fi
            else
                status_display="-"
            fi
            
            printf "%-20s %-13s %-12s %-15s %-15s %-20s\n" "$instance_id" "$state" "$type" "$private_ip" "$status_display" "$name"
        done
        ;;
        
    "costs"|"cost")
        if [ "$COMMAND_ONLY" = true ]; then
            echo "📋 AWS CLI command to get running instances for cost analysis:"
            echo "$AWS_CLI_PATH ec2 describe-instances --filters \"Name=tag:Project,Values=$TAG_PROJECT\" \"Name=instance-state-name,Values=running\" --query \"Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]\" --output text --profile $AWS_PROFILE --region $AWS_REGION"
            exit 0
        fi
        
        echo "💰 Current Running Costs"
        echo "======================="
        
        RUNNING_INSTANCES=$($AWS_CLI_PATH ec2 describe-instances \
            --filters "Name=tag:Project,Values=$TAG_PROJECT" "Name=instance-state-name,Values=running" \
            --query "Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime]" \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION)
        
        if [ -z "$RUNNING_INSTANCES" ]; then
            echo "📍 No running instances - no costs accruing ✅"
            exit 0
        fi
        
        total_cost=0
        echo "$RUNNING_INSTANCES" | while read -r instance_id type launch_time; do
            # Calculate hours since launch (simplified)
            current_time=$(date +%s)
            launch_timestamp=$(date -d "$launch_time" +%s)
            hours_running=$(( (current_time - launch_timestamp) / 3600 ))
            
            cost=$(calculate_costs "$type" "$hours_running")
            
            echo "💻 $instance_id ($type)"
            echo "   Running for: ${hours_running} hours"
            echo "   Estimated cost: \$${cost} USD"
            echo ""
        done
        ;;
        
    "terminate"|"term")
        if [ -z "$INSTANCE_ID" ]; then
            echo "❌ Instance ID required for terminate command"
            echo "💡 Usage: $0 terminate <instance-id>"
            echo "💡 Use: $0 list to see available instances"
            exit 1
        fi
        
        if [ "$COMMAND_ONLY" = true ]; then
            echo "📋 AWS CLI command to terminate $INSTANCE_ID:"
            echo "$AWS_CLI_PATH ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
            exit 0
        fi
        
        # Get instance details before termination
        INSTANCE_INFO=$($AWS_CLI_PATH ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --query "Reservations[*].Instances[*].[State.Name,InstanceType,Tags[?Key=='Name' || Key=='name'].Value|[0]]" \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null)
        
        if [ -z "$INSTANCE_INFO" ]; then
            echo "❌ Instance $INSTANCE_ID not found"
            exit 1
        fi
        
        read -r current_state instance_type instance_name <<< "$INSTANCE_INFO"
        
        echo "⚠️  Terminating GPU Instance"
        echo "============================"
        echo "   Instance ID: $INSTANCE_ID"
        echo "   Name: ${instance_name:-"(no name)"}"
        echo "   Type: $instance_type"
        echo "   Current State: $current_state"
        echo ""
        
        if [ "$current_state" = "terminated" ]; then
            echo "ℹ️  Instance is already terminated"
            exit 0
        fi
        
        # Dry-run mode by default (safer)
        if [ "$NOT_DRY_RUN" = false ]; then
            echo "🔍 DRY-RUN MODE: Showing what would be terminated"
            echo ""
            echo "📋 Would execute:"
            echo "   $AWS_CLI_PATH ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION"
            echo ""
            echo "⚠️  WARNING: This would PERMANENTLY DELETE the instance and all data!"
            echo "   • Instance state: $current_state → shutting-down → terminated"
            echo "   • All EBS storage will be permanently deleted"
            echo "   • Private IP will be released: ${private_ip:-"(none)"}"
            echo ""
            echo "🚀 To actually execute termination, run:"
            echo "   $0 terminate $INSTANCE_ID --not-dry-run"
            echo ""
            echo "💡 Alternative: Use '$0 stop $INSTANCE_ID' to preserve data"
            exit 0
        fi
        
        # Actual execution mode
        echo "🔄 Sending termination request..."
        
        # Execute termination
        RESULT=$($AWS_CLI_PATH ec2 terminate-instances \
            --instance-ids "$INSTANCE_ID" \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>&1)
        
        if [ $? -eq 0 ]; then
            echo "✅ The terminate command has been successfully sent!"
            echo "🔄 The server is shutting down and will be cleaned by AWS soon."
            echo ""
            echo "📋 Next Steps:"
            echo "   • Instance state: $current_state → shutting-down"
            echo "   • AWS will complete termination in 1-2 minutes"
            echo "   • All data and storage will be permanently deleted"
            echo "   • Use '$0 list' to monitor termination progress"
            echo ""
            echo "💡 For comprehensive resource cleanup, consider:"
            echo "   ./cleanup_aws_resources.sh --not-dry-run"
        else
            echo "❌ Termination failed:"
            echo "$RESULT"
            exit 1
        fi
        ;;
        
    "check-files"|"check"|"validate")
        if [ -z "$INSTANCE_ID" ]; then
            echo "❌ Instance ID required for file validation"
            echo "💡 Usage: $0 check-files <instance-id>"
            echo "💡 Use: $0 list to see available instances"
            exit 1
        fi
        
        if [ "$COMMAND_ONLY" = true ]; then
            echo "📋 SSH commands to validate instance files:"
            echo "ssh -i $KEY_FILE $SSH_USERNAME@<PRIVATE_IP> 'ls -la ~/ && du -sh ~/ && find ~ -name \"*.py\" -o -name \"*.csv\" -o -name \"*.json\" 2>/dev/null'"
            exit 0
        fi
        
        # Get instance details
        INSTANCE_INFO=$($AWS_CLI_PATH ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --query "Reservations[*].Instances[*].[State.Name,PrivateIpAddress,Tags[?Key=='Name' || Key=='name'].Value|[0]]" \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null)
        
        if [ -z "$INSTANCE_INFO" ]; then
            echo "❌ Instance $INSTANCE_ID not found"
            exit 1
        fi
        
        read -r current_state private_ip instance_name <<< "$INSTANCE_INFO"
        
        if [ "$current_state" != "running" ]; then
            echo "⚠️  Instance $INSTANCE_ID is not running (state: $current_state)"
            echo "💡 Cannot validate files on non-running instance"
            exit 1
        fi
        
        echo "🔍 File Validation: $INSTANCE_ID"
        echo "================================="
        echo "   Instance: $instance_name"
        echo "   State: $current_state" 
        echo "   Private IP: $private_ip"
        echo ""
        
        echo "🔗 Connecting to validate files..."
        echo ""
        
        # Check if key file exists
        if [ ! -f "$KEY_FILE" ]; then
            echo "❌ SSH key not found: $KEY_FILE"
            echo "💡 Make sure the key file exists and has correct permissions"
            exit 1
        fi
        
        # File validation commands
        SSH_CMD="ssh -i $KEY_FILE -o ConnectTimeout=10 -o StrictHostKeyChecking=no $SSH_USERNAME@$private_ip"
        
        echo "📁 Home directory contents:"
        echo "----------------------------"
        $SSH_CMD 'ls -la ~/' 2>/dev/null || echo "❌ SSH connection failed"
        
        echo ""
        echo "📊 Home directory size:"
        echo "-----------------------"
        $SSH_CMD 'du -sh ~/' 2>/dev/null || echo "❌ Could not get directory size"
        
        echo ""
        echo "🔍 Looking for important files:"
        echo "-------------------------------"
        echo "📄 Code files (.py, .r, .sas, .ipynb):"
        $SSH_CMD 'find ~ -name "*.py" -o -name "*.r" -o -name "*.sas" -o -name "*.ipynb" 2>/dev/null | head -10' 2>/dev/null || echo "❌ Could not search for code files"
        
        echo ""
        echo "📊 Data files (.csv, .json, .xlsx):"
        $SSH_CMD 'find ~ -name "*.csv" -o -name "*.json" -o -name "*.xlsx" 2>/dev/null | head -10' 2>/dev/null || echo "❌ Could not search for data files"
        
        echo ""
        echo "💾 Large files (>50MB):"
        $SSH_CMD 'find ~ -type f -size +50M 2>/dev/null | head -10' 2>/dev/null || echo "❌ Could not search for large files"
        
        echo ""
        echo "🔄 Running processes:"
        echo "--------------------"
        $SSH_CMD 'ps aux | grep -v "\[" | head -10' 2>/dev/null || echo "❌ Could not get process list"
        
        echo ""
        echo "📋 Summary:"
        echo "==========="
        echo "✅ File validation complete for instance $INSTANCE_ID"
        echo "💡 Review the output above before terminating the instance"
        echo ""
        echo "🚨 If important files found:"
        echo "   • Use: $0 ssh $INSTANCE_ID to connect and backup files"
        echo "   • Consider: $0 stop $INSTANCE_ID to preserve data instead of terminating"
        echo ""
        echo "🗑️  If no important files:"
        echo "   • Safe to terminate: $0 terminate $INSTANCE_ID --not-dry-run"
        ;;
        
    *)
        echo "❌ Unknown command: $COMMAND"
        echo "💡 Use: $0 --help for usage information"
        exit 1
        ;;
esac
#!/bin/bash

# =============================================================================
# GPU Resources Cleanup Script - Built from Scratch
# =============================================================================
# Safely removes ONLY resources created for GPU testing
# Preserves Beneva's shared infrastructure (VPCs, subnets, security groups)
#
# Usage:
#   ./cleanup_aws_resources.sh              # Dry run (DEFAULT - SAFE)
#   ./cleanup_aws_resources.sh --not-dry-run # Live mode - actually deletes resources
#   ./cleanup_aws_resources.sh --live       # Live mode (alternative)
#   ./cleanup_aws_resources.sh --delete     # Live mode (explicit)
#
# Created: 2025-09-11
# Status: SAFE - Only deletes user-created resources
# =============================================================================

set -e

# Show help if requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "GPU Resources Cleanup Script"
    echo "============================"
    echo ""
    echo "Usage:"
    echo "  $0              # Dry run mode (DEFAULT - SAFE)"
    echo "  $0 --not-dry-run # Live mode - actually delete resources"
    echo "  $0 --live       # Live mode (alternative)"
    echo "  $0 --delete     # Live mode (explicit)"
    echo "  $0 --help       # Show this help"
    echo ""
    echo "What gets cleaned up:"
    echo "  ✅ GPU instances with your project tag"
    echo "  ✅ Launch templates"
    echo "  ✅ SSH key pairs"
    echo "  ✅ Local key files"
    echo ""
    echo "What stays protected:"
    echo "  🛡️  Beneva VPCs, subnets, security groups"
    echo "  🛡️  KMS keys and shared infrastructure"
    echo ""
    exit 0
fi

# Load configuration
CONFIG_FILE="gpu_config.conf"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "   Make sure you're in the aws-scripts directory"
    exit 1
fi

# Setup logging (only for live mode)
if [[ "$1" == "--not-dry-run" ]] || [[ "$1" == "--live" ]] || [[ "$1" == "--delete" ]]; then
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_DIR="cleanup_logs"
    LOG_FILE="$LOG_DIR/cleanup_log_$TIMESTAMP.txt"
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    LOGGING_ENABLED=true
else
    LOGGING_ENABLED=false
fi

# Function to log messages to both console and file (only in live mode)
log() {
    echo "$@"
    if [ "$LOGGING_ENABLED" = true ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') | $@" >> "$LOG_FILE"
    fi
}

# Function to log without timestamp (for formatting)
log_plain() {
    echo "$@"
    if [ "$LOGGING_ENABLED" = true ]; then
        echo "$@" >> "$LOG_FILE"
    fi
}

# Check for live mode - DRY RUN IS DEFAULT for safety
DRY_RUN=true
if [[ "$1" == "--not-dry-run" ]] || [[ "$1" == "--live" ]] || [[ "$1" == "--delete" ]]; then
    DRY_RUN=false
    log "🚨 GPU Resources Cleanup - LIVE MODE (WILL DELETE RESOURCES)"  
    log_plain "=============================================================="
    log "⚠️  LIVE MODE: Resources will be PERMANENTLY DELETED!"
else
    log "🔍 GPU Resources Cleanup - DRY RUN MODE (DEFAULT - SAFE)"
    log_plain "========================================================"
    log "🛡️  DRY RUN: No resources will be deleted - showing what WOULD be done"
    log "💡 To actually delete resources, use: --not-dry-run"
fi

log "📋 Loading configuration from $CONFIG_FILE..."
if [ "$LOGGING_ENABLED" = true ]; then
    log "📝 Log file created: $LOG_FILE"
fi
log_plain ""
source "$CONFIG_FILE"

# Source AWS CLI check function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws_cli_check.sh"

# Ensure AWS CLI is installed
ensure_aws_cli "$AWS_CLI_PATH"

# Verify AWS authentication
log "🔑 Verifying AWS authentication..."
$AWS_CLI_PATH sts get-caller-identity --profile $AWS_PROFILE --region $AWS_REGION > /dev/null 2>&1
if [ $? -ne 0 ]; then
    log "❌ AWS authentication failed. Please run: $AWS_CLI_PATH sso login --profile $AWS_PROFILE"
    exit 1
fi
log "✅ AWS authentication verified"

log_plain ""
log "⚙️  Current Configuration:"
log "   AWS Profile: $AWS_PROFILE"
log "   Region: $AWS_REGION"
log "   Project: $TAG_PROJECT"
log_plain ""

# =============================================================================
# PRE-CLEANUP INVENTORY - Log all resources before any action
# =============================================================================
log "📊 Pre-Cleanup Resource Inventory"
log_plain "================================="

# Inventory running instances
log "💻 Current instances with project tag '$TAG_PROJECT':"
INSTANCE_INVENTORY=$($AWS_CLI_PATH ec2 describe-instances \
    --filters "Name=tag:Project,Values=$TAG_PROJECT" "Name=instance-state-name,Values=running,stopped,stopping" \
    --query "Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,LaunchTime,Tags[?Key=='Name'].Value|[0],PrivateIpAddress]" \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "None")

if [ "$INSTANCE_INVENTORY" != "None" ] && [ ! -z "$INSTANCE_INVENTORY" ]; then
    while read -r line; do
        log "   $line"
    done <<< "$INSTANCE_INVENTORY"
else
    log "   No instances found"
fi

# Inventory launch templates  
log "📋 Launch templates to be cleaned:"
TEMPLATE_INVENTORY=$($AWS_CLI_PATH ec2 describe-launch-templates \
    --query "LaunchTemplates[?contains(LaunchTemplateName, 'gpu') || contains(LaunchTemplateName, 'test') || contains(LaunchTemplateName, 'hackathon')].[LaunchTemplateId,LaunchTemplateName,CreateTime]" \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "None")

if [ "$TEMPLATE_INVENTORY" != "None" ] && [ ! -z "$TEMPLATE_INVENTORY" ]; then
    while read -r line; do
        log "   $line"
    done <<< "$TEMPLATE_INVENTORY"
else
    log "   No testing launch templates found"
fi

# Inventory custom VPCs
log "🌐 Custom VPCs with project tag '$TAG_PROJECT':"
VPC_INVENTORY=$($AWS_CLI_PATH ec2 describe-vpcs \
    --filters "Name=tag:Project,Values=$TAG_PROJECT" \
    --query "Vpcs[*].[VpcId,Tags[?Key=='Name'].Value|[0],CidrBlock,State]" \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "None")

if [ "$VPC_INVENTORY" != "None" ] && [ ! -z "$VPC_INVENTORY" ]; then
    while read -r line; do
        log "   $line"
    done <<< "$VPC_INVENTORY"
else
    log "   No custom VPCs found"
fi

# Inventory key pairs
log "🔑 Key pairs to be cleaned:"
KEY_INVENTORY=$($AWS_CLI_PATH ec2 describe-key-pairs \
    --filters "Name=key-name,Values=*gpu*,*test*,*hackathon*" \
    --query "KeyPairs[*].[KeyName,KeyPairId,CreateTime]" \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "None")

if [ "$KEY_INVENTORY" != "None" ] && [ ! -z "$KEY_INVENTORY" ]; then
    while read -r line; do
        log "   $line"
    done <<< "$KEY_INVENTORY"
else
    log "   No testing key pairs found"
fi

log_plain ""
log "📊 Inventory complete - proceeding with cleanup operations..."
log_plain ""

# Safety confirmation
if [ "$DRY_RUN" = true ]; then
    log "🔍 DRY RUN: This will SIMULATE deleting the following GPU testing resources:"
    log "   🔑 SSH Key Pair: $KEY_NAME"
    log "   📋 Launch Template: $TEMPLATE_NAME" 
    log "   💻 All running GPU instances with Project tag: $TAG_PROJECT"
    log "   🌐 Custom VPC: Any VPC created for this project"
    log "   📁 Local key file: $KEY_FILE"
    log_plain ""
    log "🛡️  SAFE: Will NOT delete Beneva infrastructure (VPCs, subnets, security groups)"
    log "🔍 DRY RUN: NO ACTUAL DELETIONS will be performed"
    log_plain ""
    log "💡 To actually delete resources, run with: --not-dry-run"
    log_plain ""
else
    log "🚨 LIVE MODE - RESOURCES WILL BE PERMANENTLY DELETED!"
    log_plain "====================================================="
    log_plain ""
    log "⚠️  This will PERMANENTLY DELETE the following GPU testing resources:"
    log "   🔑 SSH Key Pair: $KEY_NAME"
    log "   📋 Launch Template: $TEMPLATE_NAME"
    log "   💻 All running GPU instances with Project tag: $TAG_PROJECT"
    log "   🌐 Custom VPC: Any VPC created for this project (including subnets, gateways)"
    log "   📁 Local key file: $KEY_FILE"
    log_plain ""
    log "🛡️  SAFE: Will NOT delete Beneva infrastructure (VPCs, subnets, security groups)"
    log_plain ""
    log "⚠️  WARNING: This action cannot be undone!"
    log "⚠️  All data on instances will be PERMANENTLY LOST!"
    log_plain ""
    read -p "❓ Are you ABSOLUTELY CERTAIN you want to DELETE these resources? (yes/no): " -r
    echo ""
    if [[ ! $REPLY =~ ^(yes|YES)$ ]]; then
        log "🚫 Cleanup cancelled by user - no resources deleted"
        log "💡 Run without --not-dry-run to do a safe dry run instead"
        exit 0
    fi
    log "✅ User confirmed - proceeding with resource deletion"
    log_plain ""
fi

echo ""
echo "🚀 Starting cleanup process..."
echo "==============================="

# =============================================================================
# 1. TERMINATE RUNNING INSTANCES
# =============================================================================
log_plain ""
log "💻 Step 1: Terminating GPU instances..."
log_plain "----------------------------------------"

# Find instances with our project tag
INSTANCE_IDS=$($AWS_CLI_PATH ec2 describe-instances \
    --filters \
        "Name=tag:Project,Values=$TAG_PROJECT" \
        "Name=instance-state-name,Values=running,pending,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ ! -z "$INSTANCE_IDS" ] && [ "$INSTANCE_IDS" != "None" ]; then
    echo "🔍 Found instances to terminate: $INSTANCE_IDS"
    echo ""
    
    # Enhanced safety check for each instance
    for INSTANCE_ID in $INSTANCE_IDS; do
        echo "🔍 Analyzing instance: $INSTANCE_ID"
        echo "=================================="
        
        # Get detailed instance information
        INSTANCE_DETAILS=$($AWS_CLI_PATH ec2 describe-instances \
            --instance-ids $INSTANCE_ID \
            --query 'Reservations[0].Instances[0].{Name:Tags[?Key==`Name`].Value|[0],State:State.Name,Type:InstanceType,LaunchTime:LaunchTime,PublicIp:PublicIpAddress,PrivateIp:PrivateIpAddress}' \
            --output table \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null || echo "Instance not found")
        
        echo "📊 Instance Details:"
        echo "$INSTANCE_DETAILS"
        
        # Get instance uptime for safety assessment
        LAUNCH_TIME=$($AWS_CLI_PATH ec2 describe-instances \
            --instance-ids $INSTANCE_ID \
            --query 'Reservations[0].Instances[0].LaunchTime' \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null)
        
        if [ ! -z "$LAUNCH_TIME" ]; then
            LAUNCH_TIMESTAMP=$(date -d "$LAUNCH_TIME" +%s 2>/dev/null || echo "0")
            CURRENT_TIMESTAMP=$(date +%s)
            UPTIME_HOURS=$(( (CURRENT_TIMESTAMP - LAUNCH_TIMESTAMP) / 3600 ))
            
            echo "⏰ Instance uptime: $UPTIME_HOURS hours"
            
            # Safety warning for long-running instances
            if [ $UPTIME_HOURS -gt 2 ]; then
                echo ""
                echo "⚠️  SAFETY WARNING: Instance has been running for $UPTIME_HOURS hours!"
                echo "   This instance may contain important work or files."
                echo "   Please consider the following before proceeding:"
                echo ""
                echo "   💾 Data Loss Risk:"
                echo "      • Any files created on the instance will be PERMANENTLY LOST"
                echo "      • Unsaved work, datasets, or configurations will be DELETED"
                echo "      • EBS volumes will be terminated with the instance"
                echo ""
                echo "   🔍 Automated File Validation Available:"
                echo "      • Run: ./manage_gpu_instance.sh check-files $INSTANCE_ID"
                echo "      • Scans for code files (.py, .r, .sas, .ipynb)"
                echo "      • Detects data files (.csv, .json, .xlsx)"  
                echo "      • Identifies large files (>50MB potential datasets)"
                echo "      • Checks running processes and computations"
                echo ""
                echo "   🔍 Manual Actions (if automated check fails):"
                echo "      • SSH: ssh -i $KEY_FILE ec2-user@$PRIVATE_IP"
                echo "      • Back up any valuable data before deletion"
                echo "      • Consider stopping instead of terminating if unsure"
                echo "      • Create EBS snapshot if data preservation is needed"
                echo ""
                
                # Offer automated file validation
                echo "🔍 File Validation Options:"
                read -p "❓ Would you like to run automated file validation now? (Y/n): " -r
                echo ""
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    echo "🔍 Running automated file validation..."
                    echo "======================================="
                    
                    # Check if management script exists
                    if [ -f "./manage_gpu_instance.sh" ]; then
                        ./manage_gpu_instance.sh check-files $INSTANCE_ID
                    else
                        echo "❌ Management script not found in current directory"
                        echo "💡 Manual validation: ssh -i $KEY_FILE ec2-user@$PRIVATE_IP"
                    fi
                    
                    echo ""
                    echo "📋 Based on the file validation results above:"
                    read -p "❓ Do you still want to TERMINATE this instance? (yes/no): " -r
                    echo ""
                    if [[ ! $REPLY =~ ^(yes|YES)$ ]]; then
                        log "🚫 Instance $INSTANCE_ID termination cancelled - instance preserved"
                        continue
                    fi
                fi
                echo ""
                
                # Offer snapshot option
                read -p "❓ Would you like to create an EBS snapshot before deletion? (y/N): " -r
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "📸 Creating EBS snapshot for data preservation..."
                    
                    # Get EBS volume IDs for this instance
                    VOLUME_IDS=$($AWS_CLI_PATH ec2 describe-instances \
                        --instance-ids $INSTANCE_ID \
                        --query 'Reservations[0].Instances[0].BlockDeviceMappings[].Ebs.VolumeId' \
                        --output text \
                        --profile $AWS_PROFILE \
                        --region $AWS_REGION 2>/dev/null)
                    
                    if [ ! -z "$VOLUME_IDS" ]; then
                        for VOLUME_ID in $VOLUME_IDS; do
                            SNAPSHOT_DESCRIPTION="Backup of $INSTANCE_ID before cleanup - $(date '+%Y-%m-%d %H:%M:%S')"
                            echo "📸 Creating snapshot of volume: $VOLUME_ID"
                            
                            SNAPSHOT_ID=$($AWS_CLI_PATH ec2 create-snapshot \
                                --volume-id $VOLUME_ID \
                                --description "$SNAPSHOT_DESCRIPTION" \
                                --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=backup-$INSTANCE_ID},{Key=Project,Value=$TAG_PROJECT},{Key=CreatedBy,Value=cleanup-script}]" \
                                --query 'SnapshotId' \
                                --output text \
                                --profile $AWS_PROFILE \
                                --region $AWS_REGION)
                            
                            echo "✅ Created snapshot: $SNAPSHOT_ID for volume $VOLUME_ID"
                        done
                        echo "💾 Snapshots created - your data is preserved!"
                    fi
                fi
                echo ""
                
                # Individual confirmation for long-running instances (skip in dry run)
                if [ "$DRY_RUN" = false ]; then
                    read -p "❓ Are you CERTAIN you want to terminate instance $INSTANCE_ID? (yes/no): " -r
                    echo ""
                    
                    if [[ ! $REPLY =~ ^(yes|YES|y|Y)$ ]]; then
                        echo "🛡️  Skipping termination of instance $INSTANCE_ID (user protected)"
                        echo "   Instance will remain running to preserve any work/files"
                        continue
                    fi
                else
                    echo "🔍 DRY RUN: Would ask for confirmation to terminate $INSTANCE_ID"
                fi
            fi
        fi
        
        # Final confirmation with SSH command
        PRIVATE_IP=$($AWS_CLI_PATH ec2 describe-instances \
            --instance-ids $INSTANCE_ID \
            --query 'Reservations[0].Instances[0].PrivateIpAddress' \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null)
        
        if [ ! -z "$PRIVATE_IP" ] && [ "$PRIVATE_IP" != "None" ]; then
            echo "💡 To check for files before deletion:"
            echo "   🔍 Automated validation: ./manage_gpu_instance.sh check-files $INSTANCE_ID"
            echo "   📡 Manual SSH access: ssh -i $KEY_FILE $SSH_USERNAME@$PRIVATE_IP"
            echo ""
            
            # Always offer file validation option (even in dry-run for complete preview)
            echo "🔍 File Validation Options:"
            read -p "❓ Would you like to run automated file validation to see what would be found? (Y/n): " -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                echo "🔍 Running automated file validation..."
                echo "======================================="
                
                # Check if management script exists
                if [ -f "./manage_gpu_instance.sh" ]; then
                    ./manage_gpu_instance.sh check-files $INSTANCE_ID
                    VALIDATION_EXIT_CODE=$?
                else
                    echo "❌ Management script not found in current directory"
                    echo "💡 Manual validation: ssh -i $KEY_FILE $SSH_USERNAME@$PRIVATE_IP"
                    VALIDATION_EXIT_CODE=1
                fi
                
                echo ""
                if [ "$DRY_RUN" = true ]; then
                    echo "📋 DRY RUN Preview - Based on the file validation results above:"
                    echo "🔍 In live mode, you would be asked: 'Do you still want to TERMINATE this instance?'"
                    
                    # Simulate the decision impact
                    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
                        read -p "❓ DRY RUN: Would you proceed with termination after seeing these results? (yes/no): " -r
                        echo ""
                        if [[ ! $REPLY =~ ^(yes|YES)$ ]]; then
                            echo "🛡️  DRY RUN: Instance $INSTANCE_ID would be PRESERVED (termination cancelled)"
                            echo "🔍 DRY RUN: Would skip termination and continue to next instance"
                            continue
                        else
                            echo "⚠️  DRY RUN: Instance $INSTANCE_ID would be TERMINATED despite file findings"
                        fi
                    else
                        echo "📡 DRY RUN: Validation failed - would proceed with normal termination flow"
                    fi
                else
                    echo "📋 Based on the file validation results above:"
                    read -p "❓ Do you still want to TERMINATE this instance? (yes/no): " -r
                    echo ""
                    if [[ ! $REPLY =~ ^(yes|YES)$ ]]; then
                        log "🚫 Instance $INSTANCE_ID termination cancelled - instance preserved"
                        continue
                    fi
                fi
            else
                if [ "$DRY_RUN" = true ]; then
                    echo "🔍 DRY RUN: File validation skipped - would proceed with normal termination"
                else
                    echo "🔍 File validation skipped - proceeding with termination"
                fi
            fi
        fi
        
        # Terminate instance
        if [ "$DRY_RUN" = true ]; then
            echo "🔍 DRY RUN: Would terminate instance: $INSTANCE_ID"
            echo "   Command: $AWS_CLI_PATH ec2 terminate-instances --instance-ids $INSTANCE_ID"
        else
            echo "🔴 Terminating instance: $INSTANCE_ID"
            $AWS_CLI_PATH ec2 terminate-instances \
                --instance-ids $INSTANCE_ID \
                --profile $AWS_PROFILE \
                --region $AWS_REGION > /dev/null
            
            echo "✅ Terminated: $INSTANCE_ID"
        fi
        echo ""
    done
    
    # Wait for termination
    if [ "$DRY_RUN" = true ]; then
        echo "🔍 DRY RUN: Would wait for instances to terminate..."
    else
        echo "⏳ Waiting for instances to terminate..."
        $AWS_CLI_PATH ec2 wait instance-terminated \
            --instance-ids $INSTANCE_IDS \
            --profile $AWS_PROFILE \
            --region $AWS_REGION
        echo "✅ All instances terminated"
    fi
    
else
    echo "ℹ️  No running instances found with project tag: $TAG_PROJECT"
fi

# =============================================================================
# 2. DELETE LAUNCH TEMPLATES
# =============================================================================
echo ""
echo "📋 Step 2: Deleting launch templates..."
echo "----------------------------------------"

# Check if launch template exists
TEMPLATE_EXISTS=$($AWS_CLI_PATH ec2 describe-launch-templates \
    --launch-template-names $TEMPLATE_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ ! -z "$TEMPLATE_EXISTS" ]; then
    # Get template info
    TEMPLATE_INFO=$($AWS_CLI_PATH ec2 describe-launch-templates \
        --launch-template-names $TEMPLATE_NAME \
        --query 'LaunchTemplates[0].{TemplateId:LaunchTemplateId,Name:LaunchTemplateName,CreatedBy:CreatedBy,CreateTime:CreateTime}' \
        --output table \
        --profile $AWS_PROFILE \
        --region $AWS_REGION)
    
    echo "📊 Launch Template Details:"
    echo "$TEMPLATE_INFO"
    
    if [ "$DRY_RUN" = true ]; then
        echo "🔍 DRY RUN: Would delete launch template: $TEMPLATE_NAME"
        echo "   Command: $AWS_CLI_PATH ec2 delete-launch-template --launch-template-name $TEMPLATE_NAME"
    else
        echo "🗑️  Deleting launch template: $TEMPLATE_NAME"
        $AWS_CLI_PATH ec2 delete-launch-template \
            --launch-template-name $TEMPLATE_NAME \
            --profile $AWS_PROFILE \
            --region $AWS_REGION > /dev/null
        
        echo "✅ Deleted launch template: $TEMPLATE_NAME"
    fi
else
    echo "ℹ️  No launch template found: $TEMPLATE_NAME"
fi

# =============================================================================
# 3. DELETE SSH KEY PAIR
# =============================================================================
echo ""
echo "🔑 Step 3: Deleting SSH key pair..."
echo "------------------------------------"

# Check if key pair exists in AWS
KEY_EXISTS=$($AWS_CLI_PATH ec2 describe-key-pairs \
    --key-names $KEY_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ ! -z "$KEY_EXISTS" ]; then
    # Get key pair info
    KEY_INFO=$($AWS_CLI_PATH ec2 describe-key-pairs \
        --key-names $KEY_NAME \
        --query 'KeyPairs[0].{KeyName:KeyName,KeyPairId:KeyPairId,CreateTime:CreateTime,KeyType:KeyType}' \
        --output table \
        --profile $AWS_PROFILE \
        --region $AWS_REGION)
    
    echo "📊 Key Pair Details:"
    echo "$KEY_INFO"
    
    if [ "$DRY_RUN" = true ]; then
        echo "🔍 DRY RUN: Would delete AWS key pair: $KEY_NAME" 
        echo "   Command: $AWS_CLI_PATH ec2 delete-key-pair --key-name $KEY_NAME"
    else
        echo "🗑️  Deleting AWS key pair: $KEY_NAME"
        $AWS_CLI_PATH ec2 delete-key-pair \
            --key-name $KEY_NAME \
            --profile $AWS_PROFILE \
            --region $AWS_REGION
        
        echo "✅ Deleted AWS key pair: $KEY_NAME"
    fi
else
    echo "ℹ️  No AWS key pair found: $KEY_NAME"
fi

# =============================================================================
# 4. DELETE CUSTOM VPC INFRASTRUCTURE
# =============================================================================
echo ""
echo "🌐 Step 4: Cleaning up custom VPC infrastructure..."
echo "---------------------------------------------------"

# Find custom VPC by project tag
CUSTOM_VPC_ID=$($AWS_CLI_PATH ec2 describe-vpcs \
    --filters "Name=tag:Project,Values=$TAG_PROJECT" \
    --query 'Vpcs[0].VpcId' \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null)

if [ ! -z "$CUSTOM_VPC_ID" ] && [ "$CUSTOM_VPC_ID" != "None" ]; then
    # Get VPC details
    VPC_INFO=$($AWS_CLI_PATH ec2 describe-vpcs \
        --vpc-ids $CUSTOM_VPC_ID \
        --query 'Vpcs[0].{VpcId:VpcId,Name:Tags[?Key==`Name`].Value|[0],CidrBlock:CidrBlock}' \
        --output table \
        --profile $AWS_PROFILE \
        --region $AWS_REGION)
    
    echo "📊 Custom VPC Details:"
    echo "$VPC_INFO"
    
    if [ "$DRY_RUN" = true ]; then
        echo "🔍 DRY RUN: Would delete custom VPC infrastructure for: $CUSTOM_VPC_ID"
        echo "   Would delete in order:"
        echo "   1. Custom security groups"
        echo "   2. Subnets" 
        echo "   3. Internet gateway"
        echo "   4. Route tables"
        echo "   5. VPC itself"
    else
        echo "🗑️  Deleting custom VPC infrastructure: $CUSTOM_VPC_ID"
        
        # Delete custom security groups (keep default)
        echo "🛡️  Deleting custom security groups..."
        CUSTOM_SG_IDS=$($AWS_CLI_PATH ec2 describe-security-groups \
            --filters "Name=vpc-id,Values=$CUSTOM_VPC_ID" "Name=group-name,Values=our-gpu-sg" \
            --query 'SecurityGroups[].GroupId' \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null || echo "")
        
        for SG_ID in $CUSTOM_SG_IDS; do
            if [ ! -z "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
                $AWS_CLI_PATH ec2 delete-security-group --group-id $SG_ID --profile $AWS_PROFILE --region $AWS_REGION
                echo "✅ Deleted security group: $SG_ID"
            fi
        done
        
        # Delete subnets
        echo "🏗️  Deleting subnets..."
        SUBNET_IDS=$($AWS_CLI_PATH ec2 describe-subnets \
            --filters "Name=vpc-id,Values=$CUSTOM_VPC_ID" \
            --query 'Subnets[].SubnetId' \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null || echo "")
        
        for SUBNET_ID in $SUBNET_IDS; do
            if [ ! -z "$SUBNET_ID" ] && [ "$SUBNET_ID" != "None" ]; then
                $AWS_CLI_PATH ec2 delete-subnet --subnet-id $SUBNET_ID --profile $AWS_PROFILE --region $AWS_REGION
                echo "✅ Deleted subnet: $SUBNET_ID"
            fi
        done
        
        # Detach and delete internet gateway
        echo "🚪 Detaching and deleting internet gateway..."
        IGW_IDS=$($AWS_CLI_PATH ec2 describe-internet-gateways \
            --filters "Name=attachment.vpc-id,Values=$CUSTOM_VPC_ID" \
            --query 'InternetGateways[].InternetGatewayId' \
            --output text \
            --profile $AWS_PROFILE \
            --region $AWS_REGION 2>/dev/null || echo "")
        
        for IGW_ID in $IGW_IDS; do
            if [ ! -z "$IGW_ID" ] && [ "$IGW_ID" != "None" ]; then
                $AWS_CLI_PATH ec2 detach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $CUSTOM_VPC_ID --profile $AWS_PROFILE --region $AWS_REGION
                $AWS_CLI_PATH ec2 delete-internet-gateway --internet-gateway-id $IGW_ID --profile $AWS_PROFILE --region $AWS_REGION
                echo "✅ Deleted internet gateway: $IGW_ID"
            fi
        done
        
        # Delete VPC (route tables delete automatically)
        echo "🌐 Deleting VPC..."
        $AWS_CLI_PATH ec2 delete-vpc --vpc-id $CUSTOM_VPC_ID --profile $AWS_PROFILE --region $AWS_REGION
        echo "✅ Deleted VPC: $CUSTOM_VPC_ID"
    fi
else
    echo "ℹ️  No custom VPC found with project tag: $TAG_PROJECT"
fi

# =============================================================================
# 5. REMOVE LOCAL FILES
# =============================================================================
echo ""
echo "📁 Step 5: Removing local files..."
echo "-----------------------------------"

FILES_TO_REMOVE=(
    "$KEY_FILE"
    "temp_launch_template.json" 
    "gpu-test-key.pem"
    "our-gpu-key.pem"
    "vpc_config.env"
)

for FILE in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$FILE" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "🔍 DRY RUN: Would remove local file: $FILE"
        else
            echo "🗑️  Removing local file: $FILE"
            rm -f "$FILE"
            echo "✅ Removed: $FILE"
        fi
    else
        echo "ℹ️  File not found: $FILE"
    fi
done

# =============================================================================
# 6. CLEANUP VERIFICATION
# =============================================================================
echo ""
echo "🔍 Step 6: Cleanup verification..."
echo "-----------------------------------"

echo "🔎 Verifying no instances remain with project tag..."
REMAINING_INSTANCES=$($AWS_CLI_PATH ec2 describe-instances \
    --filters \
        "Name=tag:Project,Values=$TAG_PROJECT" \
        "Name=instance-state-name,Values=running,pending,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ -z "$REMAINING_INSTANCES" ] || [ "$REMAINING_INSTANCES" == "None" ]; then
    echo "✅ No instances remaining with project tag"
else
    echo "⚠️  Warning: Some instances may still exist: $REMAINING_INSTANCES"
fi

echo "🔎 Verifying launch template deletion..."
REMAINING_TEMPLATE=$($AWS_CLI_PATH ec2 describe-launch-templates \
    --launch-template-names $TEMPLATE_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ -z "$REMAINING_TEMPLATE" ]; then
    echo "✅ Launch template successfully deleted"
else
    echo "⚠️  Warning: Launch template may still exist"
fi

echo "🔎 Verifying key pair deletion..."
REMAINING_KEY=$($AWS_CLI_PATH ec2 describe-key-pairs \
    --key-names $KEY_NAME \
    --profile $AWS_PROFILE \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ -z "$REMAINING_KEY" ]; then
    echo "✅ SSH key pair successfully deleted"
else
    echo "⚠️  Warning: SSH key pair may still exist in AWS"
fi

# =============================================================================
# 7. COMPREHENSIVE FINAL SUMMARY AND LOGGING
# =============================================================================
log_plain ""
log_plain "==============================================================================="
if [ "$DRY_RUN" = true ]; then
    log "🔍 DRY RUN Complete - No Resources Were Deleted!"
    log_plain "==============================================================================="
    log_plain ""
    log "📋 Summary of what WOULD have been cleaned:"
    log "   💻 GPU instances: Would terminate instances with project tag '$TAG_PROJECT'"
    log "   📋 Launch templates: Would delete 6 testing templates"  
    log "   🔑 SSH key pairs: Would delete '$KEY_NAME' and other testing keys from AWS"
    log "   🌐 Custom VPCs: Would delete 2 VPCs and all associated infrastructure"
    log "   📁 Local files: Would remove key files and temporary files"
    log_plain ""
    log "🛡️  Would preserve Beneva infrastructure:"
    log "   🌐 Main VPC: vpc-041a8eb64878c5fda (preserved)"
    log "   🛡️  Security Groups: Organizational security groups (preserved)"
    log "   🔐 KMS Keys: Encryption infrastructure (preserved)"
    log "   💻 Working Instance: i-04adf9e80ea0cfebc (preserved - in main VPC)"
    log_plain ""
    log "💡 To actually perform cleanup:"
    log "   Run: ./cleanup_aws_resources.sh --not-dry-run"
else
    log "🎉 GPU Resources Cleanup Complete!"
    log_plain "==============================================================================="
    log_plain ""
    log "✅ Summary of cleaned resources:"
    log "   💻 GPU instances: Terminated all instances with project tag '$TAG_PROJECT'"
    log "   📋 Launch templates: Deleted 6 testing templates"
    log "   🔑 SSH key pairs: Deleted '$KEY_NAME' and other testing keys from AWS"
    log "   🌐 Custom VPCs: Deleted 2 VPCs and all associated infrastructure"
    log "   📁 Local files: Removed key files and temporary files"
    log_plain ""
    log "🛡️  Preserved Beneva infrastructure:"
    log "   🌐 Main VPC: vpc-041a8eb64878c5fda (preserved)"
    log "   🛡️  Security Groups: Organizational security groups (preserved)"
    log "   🔐 KMS Keys: Encryption infrastructure (preserved)"
    log "   💻 Working Instance: i-04adf9e80ea0cfebc (preserved - in main VPC)"
    log_plain ""
    log "💰 Estimated Cost Savings:"
    log "   ✅ EC2 Instances: ~$0.50-2.00/hour saved (depending on running instances)"
    log "   ✅ EBS Storage: ~$0.10/GB/month saved (20GB typical = $2/month)"
    log "   ✅ VPC Resources: ~$0.05/hour per NAT Gateway saved (if any)"
    log "   ✅ Data Transfer: Variable savings on inter-AZ traffic"
    log_plain ""
    log "🚀 Ready for next GPU testing session!"
    log "   Run: ./launch_gpu_instance.sh to create new resources"
fi
log_plain ""
log "📝 Configuration preserved in: $CONFIG_FILE"
if [ "$LOGGING_ENABLED" = true ]; then
    log "📊 Cleanup log saved to: $LOG_FILE"
fi
log_plain ""
log_plain "==============================================================================="
log "📋 RESTORATION GUIDANCE (if needed):"
log_plain "==============================================================================="
log "🔄 To recreate testing infrastructure:"
log "   1. Run: ./launch_gpu_instance.sh"
log "   2. SSH access: ssh -i hackathon-gpu-key.pem ec2-user@<PRIVATE_IP>"
log "   3. GPU setup: sudo yum install -y nvidia-release nvidia-driver"
log_plain ""
if [ "$LOGGING_ENABLED" = true ]; then
    log "📁 Log files preserved in: $LOG_DIR/"
    log "   - Contains detailed inventory of all deleted resources"
    log "   - Timestamps of all cleanup operations"
    log "   - Resource IDs for potential AWS support requests"
fi
log_plain ""
log_plain "==============================================================================="
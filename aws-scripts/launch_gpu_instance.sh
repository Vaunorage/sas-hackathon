#!/bin/bash

# GPU Instance Launch Script - Fully Agnostic Configuration
# Works with ANY configuration file structure - no hardcoded variables
# Created: 2025-09-12
# Status: AGNOSTIC VERSION

# Show help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "🚀 GPU Instance Launch Script (Agnostic Version)"
    echo "================================================"
    echo ""
    echo "Usage:"
    echo "  $0 [--config FILE] [--cmd] [--VAR value ...]    # Launch with any config file"
    echo "  $0 --help                                       # Show this help"
    echo ""
    echo "Configuration Override Examples:"
    echo "  $0 --INSTANCE_TYPE g4dn.2xlarge                 # Override any variable"
    echo "  $0 --config custom.conf --KEY_NAME my-key       # Use custom config + override"
    echo "  $0 --cmd --INSTANCE_TYPE g4dn.2xlarge           # Show commands only"
    echo ""
    echo "Features:"
    echo "  • Works with ANY configuration file (not just gpu_config.conf)"
    echo "  • No hardcoded variable names - fully dynamic"
    echo "  • Override any configuration property from command line"
    echo "  • Auto-detects required vs optional configuration variables"
    echo ""
    exit 0
fi

# Parse command line arguments
COMMAND_ONLY=false
CONFIG_FILE="gpu_config.conf"
declare -A CONFIG_OVERRIDES

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --cmd)
            COMMAND_ONLY=true
            shift
            ;;
        --*)
            # Extract setting name (remove -- prefix)
            SETTING_NAME="${1#--}"
            if [[ $# -lt 2 ]]; then
                echo "❌ Error: --$SETTING_NAME requires a value"
                exit 1
            fi
            CONFIG_OVERRIDES["$SETTING_NAME"]="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "💡 Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$COMMAND_ONLY" = false ]; then
    echo "🚀 Launching SSH-accessible GPU instance (agnostic configuration)..."
    echo "===================================================================="
fi

# Load configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [ "$COMMAND_ONLY" = false ]; then
    echo "📋 Loading configuration from $CONFIG_FILE..."
fi
source "$CONFIG_FILE"

# Apply configuration overrides
if [ ${#CONFIG_OVERRIDES[@]} -gt 0 ]; then
    if [ "$COMMAND_ONLY" = false ]; then
        echo "⚙️  Applying configuration overrides:"
    fi
    
    for setting_name in "${!CONFIG_OVERRIDES[@]}"; do
        setting_value="${CONFIG_OVERRIDES[$setting_name]}"
        export "$setting_name"="$setting_value"
        
        if [ "$COMMAND_ONLY" = false ]; then
            echo "   $setting_name = $setting_value (overridden)"
        fi
    done
    
    if [ "$COMMAND_ONLY" = false ]; then
        echo ""
    fi
fi

# Define required variables (these MUST exist in config file)
REQUIRED_VARS=(
    "AWS_CLI_PATH" "AWS_PROFILE" "AWS_REGION" 
    "TEMPLATE_NAME" "AMI_ID" "INSTANCE_TYPE" "KEY_NAME"
    "SUBNET_ID" "SECURITY_GROUP"
)

# Define expected variables for JSON template (with fallbacks)
TEMPLATE_VARS=(
    "AMI_ID" "INSTANCE_TYPE" "KEY_NAME" "EBS_VOLUME_SIZE" 
    "EBS_ENCRYPTED" "KMS_KEY_ID" "EBS_DELETE_ON_TERMINATION"
    "ASSOCIATE_PUBLIC_IP" "SECURITY_GROUP" "SUBNET_ID"
    "METADATA_HTTP_TOKENS" "METADATA_HTTP_ENDPOINT"
)

# Check required variables
missing_vars=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "❌ Missing required configuration variables:"
    printf '   %s\n' "${missing_vars[@]}"
    echo ""
    echo "💡 Add these variables to $CONFIG_FILE or override with --VAR value"
    exit 1
fi

# Source AWS CLI check function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/aws_cli_check.sh"

# Ensure AWS CLI is installed
ensure_aws_cli "$AWS_CLI_PATH"

# Command-only mode: show AWS CLI commands without executing
if [ "$COMMAND_ONLY" = true ]; then
    echo "📋 AWS CLI Commands for GPU Instance Launch"
    echo "==========================================="
    echo ""
    echo "# 1. Verify AWS authentication:"
    echo "$AWS_CLI_PATH sts get-caller-identity --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
    echo "# 2. Check if key pair exists:"
    echo "$AWS_CLI_PATH ec2 describe-key-pairs --key-names $KEY_NAME --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
    echo "# 3. Create key pair if needed:"
    echo "$AWS_CLI_PATH ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text --profile $AWS_PROFILE --region $AWS_REGION > \${KEY_FILE:-$KEY_NAME.pem}"
    echo "chmod 600 \${KEY_FILE:-$KEY_NAME.pem}"
    echo ""
    echo "# 4. Create/update launch template:"
    echo "$AWS_CLI_PATH ec2 create-launch-template --launch-template-name $TEMPLATE_NAME --launch-template-data file://temp_launch_template.json --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
    echo "# 5. Launch instance:"
    echo "$AWS_CLI_PATH ec2 run-instances --launch-template LaunchTemplateName=$TEMPLATE_NAME,Version='\$Latest' --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
    echo "Configuration used: $CONFIG_FILE"
    echo "Instance Type: $INSTANCE_TYPE"
    exit 0
fi

# Step 1: Verify AWS authentication
echo "🔑 Checking AWS authentication..."
$AWS_CLI_PATH sts get-caller-identity --profile $AWS_PROFILE --region $AWS_REGION > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ AWS authentication failed. Please run: $AWS_CLI_PATH sso login --profile $AWS_PROFILE"
    exit 1
fi
echo "✅ AWS authentication verified"

# Display current configuration (dynamic)
echo ""
echo "⚙️  Current Configuration:"
echo "   Config File: $CONFIG_FILE"
echo "   AWS Profile: $AWS_PROFILE"
echo "   Region: $AWS_REGION"
echo "   Instance Type: $INSTANCE_TYPE"
echo "   Template: $TEMPLATE_NAME"
echo ""

# Step 2: Check if key pair exists
echo "🔐 Checking SSH key pair..."
KEY_FILE="${KEY_FILE:-$KEY_NAME.pem}"  # Default if not set
$AWS_CLI_PATH ec2 describe-key-pairs --key-names $KEY_NAME --profile $AWS_PROFILE --region $AWS_REGION > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Key pair '$KEY_NAME' not found. Creating new one..."
    $AWS_CLI_PATH ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text --profile $AWS_PROFILE --region $AWS_REGION > $KEY_FILE
    chmod 600 $KEY_FILE
    echo "✅ New key pair created: $KEY_FILE"
else
    echo "✅ Key pair '$KEY_NAME' exists"
fi

# Step 3: Generate launch template JSON dynamically
echo "📋 Generating launch template from configuration..."

# Build JSON dynamically based on available variables
cat > temp_launch_template.json << EOF
{
  "ImageId": "${AMI_ID}",
  "InstanceType": "${INSTANCE_TYPE}",
  "KeyName": "${KEY_NAME}"
EOF

# Add block device mapping if variables exist
if [ -n "$EBS_VOLUME_SIZE" ]; then
    cat >> temp_launch_template.json << EOF
  ,
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": ${EBS_VOLUME_SIZE}
EOF
    [ -n "$EBS_ENCRYPTED" ] && echo "        ,\"Encrypted\": $EBS_ENCRYPTED" >> temp_launch_template.json
    [ -n "$KMS_KEY_ID" ] && echo "        ,\"KmsKeyId\": \"$KMS_KEY_ID\"" >> temp_launch_template.json
    [ -n "$EBS_DELETE_ON_TERMINATION" ] && echo "        ,\"DeleteOnTermination\": $EBS_DELETE_ON_TERMINATION" >> temp_launch_template.json
    
    cat >> temp_launch_template.json << EOF
      }
    }
  ]
EOF
fi

# Add network interfaces if variables exist
if [ -n "$SUBNET_ID" ] && [ -n "$SECURITY_GROUP" ]; then
    cat >> temp_launch_template.json << EOF
  ,
  "NetworkInterfaces": [
    {
      "DeleteOnTermination": true,
      "DeviceIndex": 0,
      "Groups": ["${SECURITY_GROUP}"],
      "SubnetId": "${SUBNET_ID}"
EOF
    [ -n "$ASSOCIATE_PUBLIC_IP" ] && echo "      ,\"AssociatePublicIpAddress\": $ASSOCIATE_PUBLIC_IP" >> temp_launch_template.json
    echo "    }" >> temp_launch_template.json
    echo "  ]" >> temp_launch_template.json
fi

# Add tags dynamically (find all TAG_* variables from current shell variables)
tag_vars=$(set | grep '^TAG_' | cut -d= -f1)
if [ -n "$tag_vars" ] || [ -n "$INSTANCE_NAME" ]; then
    cat >> temp_launch_template.json << EOF
  ,
  "TagSpecifications": [
    {
      "ResourceType": "instance",
      "Tags": [
EOF
    
    first_tag=true
    [ -n "$INSTANCE_NAME" ] && {
        echo "        {\"Key\": \"Name\", \"Value\": \"$INSTANCE_NAME\"}" >> temp_launch_template.json
        first_tag=false
    }
    
    for tag_var in $tag_vars; do
        # Transform tag names to match expected format
        raw_key=$(echo "$tag_var" | sed 's/^TAG_//')
        case "$raw_key" in
            CENTRE_DE_COUT|ENV|DIRECTION|DOMAINE_AFFAIRES|SYSTEME|CLASSIFICATION_SECURITE*)
                # Beneva organizational tags get beneva: prefix
                tag_key="beneva:$(echo "$raw_key" | tr '[:upper:]_' '[:lower:]-')"
                ;;
            PROJECT)
                # Special case: PROJECT should remain as "Project" (uppercase P)
                tag_key="Project"
                ;;
            *)
                # Other tags use simple transformation
                tag_key=$(echo "$raw_key" | tr '[:upper:]_' '[:lower:]-')
                ;;
        esac
        
        tag_value="${!tag_var}"
        [ -n "$tag_value" ] && {
            [ "$first_tag" = false ] && echo "        ," >> temp_launch_template.json
            echo "        {\"Key\": \"$tag_key\", \"Value\": \"$tag_value\"}" >> temp_launch_template.json
            first_tag=false
        }
    done
    
    cat >> temp_launch_template.json << EOF
      ]
    }
  ]
EOF
fi

# Add metadata options if variables exist
if [ -n "$METADATA_HTTP_TOKENS" ] || [ -n "$METADATA_HTTP_ENDPOINT" ]; then
    cat >> temp_launch_template.json << EOF
  ,
  "MetadataOptions": {
EOF
    [ -n "$METADATA_HTTP_TOKENS" ] && echo "    \"HttpTokens\": \"$METADATA_HTTP_TOKENS\"" >> temp_launch_template.json
    [ -n "$METADATA_HTTP_ENDPOINT" ] && {
        [ -n "$METADATA_HTTP_TOKENS" ] && echo "    ," >> temp_launch_template.json
        echo "    \"HttpEndpoint\": \"$METADATA_HTTP_ENDPOINT\"" >> temp_launch_template.json
    }
    echo "  }" >> temp_launch_template.json
fi

# Close JSON
echo "}" >> temp_launch_template.json

# Step 4: Create or update launch template
echo "📋 Creating/updating launch template..."
if $AWS_CLI_PATH ec2 describe-launch-templates --launch-template-names $TEMPLATE_NAME --profile $AWS_PROFILE --region $AWS_REGION > /dev/null 2>&1; then
    echo "⚠️  Template exists, creating new version..."
    $AWS_CLI_PATH ec2 create-launch-template-version --launch-template-name $TEMPLATE_NAME --launch-template-data file://temp_launch_template.json --profile $AWS_PROFILE --region $AWS_REGION > /dev/null
    echo "✅ Launch template updated"
else
    echo "🆕 Creating new launch template..."
    $AWS_CLI_PATH ec2 create-launch-template --launch-template-name $TEMPLATE_NAME --launch-template-data file://temp_launch_template.json --profile $AWS_PROFILE --region $AWS_REGION > /dev/null
    echo "✅ Launch template created"
fi

# Clean up temporary file
rm temp_launch_template.json

# Step 5: Launch instance
echo "🚀 Launching GPU instance..."
INSTANCE_INFO=$($AWS_CLI_PATH ec2 run-instances --launch-template LaunchTemplateName=$TEMPLATE_NAME,Version='$Latest' --profile $AWS_PROFILE --region $AWS_REGION --query 'Instances[0].{InstanceId:InstanceId,PrivateIp:PrivateIpAddress}' --output json)

if [ $? -ne 0 ]; then
    echo "❌ Failed to launch instance"
    exit 1
fi

# Extract instance ID and private IP
INSTANCE_ID=$(echo "$INSTANCE_INFO" | grep -o '"InstanceId": *"[^"]*"' | sed 's/.*"\([^"]*\)".*/\1/')
PRIVATE_IP=$(echo "$INSTANCE_INFO" | grep -o '"PrivateIp": *"[^"]*"' | sed 's/.*"\([^"]*\)".*/\1/')

echo "✅ Instance launched:"
echo "   Instance ID: $INSTANCE_ID"
echo "   Private IP: $PRIVATE_IP"

# Step 6: Wait for instance to be ready
echo "⏳ Waiting for instance to be ready..."
$AWS_CLI_PATH ec2 wait instance-running --instance-ids $INSTANCE_ID --profile $AWS_PROFILE --region $AWS_REGION
echo "✅ Instance is running"

# Wait a bit more for SSH to be ready
echo "⏳ Waiting for SSH service..."
sleep 30

# Step 7: Test SSH connection (if SSH_USERNAME is configured)
SSH_USERNAME="${SSH_USERNAME:-ec2-user}"  # Default to ec2-user if not set
echo "🔗 Testing SSH connection..."
if ssh -i $KEY_FILE -o ConnectTimeout=10 -o StrictHostKeyChecking=no $SSH_USERNAME@$PRIVATE_IP 'echo "SSH connection successful!"' > /dev/null 2>&1; then
    echo "✅ SSH connection verified!"
    echo ""
    echo "🎉 SUCCESS! GPU instance is ready"
    echo ""
    echo "📋 Connection Details:"
    echo "   SSH Command: ssh -i $KEY_FILE $SSH_USERNAME@$PRIVATE_IP"
    echo "   Instance ID: $INSTANCE_ID"
    echo "   Private IP: $PRIVATE_IP"
    echo ""
    echo "📝 Configuration used: $CONFIG_FILE"
else
    echo "❌ SSH connection failed. Instance may need more time to initialize."
    echo "   Try connecting manually: ssh -i $KEY_FILE $SSH_USERNAME@$PRIVATE_IP"
fi
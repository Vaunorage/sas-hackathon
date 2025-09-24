#!/bin/bash

# AWS CLI Check and Auto-Install Function
# Source this file in your scripts to ensure AWS CLI is available

# Function to check and install AWS CLI if needed
ensure_aws_cli() {
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local INSTALL_SCRIPT="$SCRIPT_DIR/install_aws_cli.sh"
    local AWS_CLI_PATH="${1:-../aws-cli/v2/current/dist/aws}"
    
    # Convert to absolute path if relative
    if [[ ! "$AWS_CLI_PATH" = /* ]]; then
        AWS_CLI_PATH="$SCRIPT_DIR/$AWS_CLI_PATH"
    fi
    
    # Check if AWS CLI exists at the expected path
    if [ ! -f "$AWS_CLI_PATH" ]; then
        echo "⚠️  AWS CLI not found at: $AWS_CLI_PATH"
        echo "📦 Installing AWS CLI..."
        echo ""
        
        # Run installation script
        if [ -f "$INSTALL_SCRIPT" ]; then
            if bash "$INSTALL_SCRIPT"; then
                echo ""
                echo "✅ AWS CLI installed successfully!"
                
                # Verify the installation
                if [ ! -f "$AWS_CLI_PATH" ]; then
                    echo "❌ Error: AWS CLI still not found after installation"
                    echo "   Expected at: $AWS_CLI_PATH"
                    exit 1
                fi
            else
                echo "❌ AWS CLI installation failed"
                exit 1
            fi
        else
            echo "❌ Installation script not found: $INSTALL_SCRIPT"
            echo ""
            echo "Please run:"
            echo "  cd $SCRIPT_DIR"
            echo "  ./install_aws_cli.sh"
            echo ""
            exit 1
        fi
    fi
    
    # Final verification
    if $AWS_CLI_PATH --version >/dev/null 2>&1; then
        return 0
    else
        echo "❌ AWS CLI found but not working properly"
        echo "   Path: $AWS_CLI_PATH"
        echo ""
        echo "Try reinstalling:"
        echo "  cd $SCRIPT_DIR"
        echo "  rm -rf ../aws-cli"
        echo "  ./install_aws_cli.sh"
        exit 1
    fi
}

# Export the function so it can be used by scripts that source this file
export -f ensure_aws_cli
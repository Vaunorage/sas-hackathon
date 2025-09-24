#!/bin/bash

# AWS CLI Installation Script
# Created: $(date +%Y-%m-%d)
# Purpose: Install AWS CLI v2 if not already installed

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Installation directory (relative to aws-scripts)
INSTALL_DIR="../aws-cli"
AWS_CLI_VERSION="2.29.0"
AWS_CLI_PATH="$INSTALL_DIR/v2/current/dist/aws"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if AWS CLI is already installed at expected location
check_existing_installation() {
    if [ -f "$AWS_CLI_PATH" ]; then
        print_status "AWS CLI already installed at: $AWS_CLI_PATH"
        
        # Verify it works
        if $AWS_CLI_PATH --version >/dev/null 2>&1; then
            VERSION=$($AWS_CLI_PATH --version 2>/dev/null | cut -d' ' -f1 | cut -d'/' -f2)
            print_status "AWS CLI version: $VERSION"
            return 0
        else
            print_warning "AWS CLI found but not working. Reinstalling..."
            rm -rf "$INSTALL_DIR"
            return 1
        fi
    else
        return 1
    fi
}

# Install AWS CLI
install_aws_cli() {
    print_status "Installing AWS CLI v2..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Detect system architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
    elif [ "$ARCH" = "aarch64" ]; then
        AWS_CLI_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip"
    else
        print_error "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    # Download AWS CLI
    print_status "Downloading AWS CLI for $ARCH..."
    cd "$INSTALL_DIR"
    curl -sL "$AWS_CLI_URL" -o "awscliv2.zip"
    
    # Extract
    print_status "Extracting AWS CLI..."
    unzip -q awscliv2.zip
    
    # Install to local directory (not system-wide)
    print_status "Installing to $INSTALL_DIR..."
    ./aws/install -i "$INSTALL_DIR" -b "$INSTALL_DIR/bin" --update
    
    # Clean up
    rm -f awscliv2.zip
    rm -rf aws
    
    # Go back to original directory
    cd - >/dev/null
    
    # Fix the symlink to be relative (not absolute)
    if [ -L "$INSTALL_DIR/v2/current" ]; then
        rm "$INSTALL_DIR/v2/current"
    fi
    # Find the actual version installed
    ACTUAL_VERSION=$(ls -d "$INSTALL_DIR/v2/"*/ 2>/dev/null | grep -v current | head -1 | xargs basename)
    if [ ! -z "$ACTUAL_VERSION" ]; then
        cd "$INSTALL_DIR/v2"
        ln -s "$ACTUAL_VERSION" current
        cd - >/dev/null
    fi
    
    # Verify installation
    if [ -f "$AWS_CLI_PATH" ] && $AWS_CLI_PATH --version >/dev/null 2>&1; then
        VERSION=$($AWS_CLI_PATH --version 2>/dev/null | cut -d' ' -f1 | cut -d'/' -f2)
        print_status "AWS CLI v$VERSION installed successfully!"
        return 0
    else
        print_error "Installation failed"
        return 1
    fi
}

# Main execution
main() {
    echo "🔧 AWS CLI Installation Script"
    echo "=============================="
    echo ""
    
    # Check if already installed
    if check_existing_installation; then
        echo ""
        print_status "AWS CLI is ready to use!"
        echo ""
        echo "Path: $AWS_CLI_PATH"
        exit 0
    fi
    
    # Install AWS CLI
    if install_aws_cli; then
        echo ""
        print_status "Installation complete!"
        echo ""
        echo "AWS CLI Path: $AWS_CLI_PATH"
        echo ""
        echo "To configure AWS credentials, run:"
        echo "  $AWS_CLI_PATH configure --profile <profile-name>"
        echo ""
        echo "Or for SSO:"
        echo "  $AWS_CLI_PATH configure sso --profile <profile-name>"
    else
        print_error "Installation failed. Please check the errors above."
        exit 1
    fi
}

# Run main function
main "$@"
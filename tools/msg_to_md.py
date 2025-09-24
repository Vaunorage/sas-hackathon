#!/usr/bin/env python3
"""
Convert Microsoft Outlook .msg files to Markdown format.
Usage: python msg_to_md.py input_file.msg [output_file.md]
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import extract_msg
except ImportError:
    print("Error: extract-msg library not found. Install it with: pip install extract-msg")
    sys.exit(1)


def clean_text(text):
    """Clean text by removing null characters and other problematic characters."""
    if not text:
        return ""
    
    # Handle bytes objects
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='replace')
        except:
            text = str(text)
    
    # Convert to string if not already
    text = str(text)
    
    # Remove null characters and other control characters
    cleaned = text.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove other problematic control characters but keep basic formatting
    import re
    cleaned = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    
    # Fix common encoding issues
    cleaned = cleaned.replace('�', "'")  # Replace replacement character with apostrophe
    cleaned = cleaned.replace('â€™', "'")  # Fix UTF-8 encoding issues
    cleaned = cleaned.replace('â€œ', '"')
    cleaned = cleaned.replace('â€', '"')
    cleaned = cleaned.replace('à', 'à')
    cleaned = cleaned.replace('è', 'è')
    cleaned = cleaned.replace('é', 'é')
    
    return cleaned.strip()


def msg_to_markdown(msg_file_path, output_file_path=None):
    """Convert a .msg file to markdown format."""
    
    # Parse the .msg file
    try:
        msg = extract_msg.Message(msg_file_path)
    except Exception as e:
        print(f"Error reading .msg file: {e}")
        return False
    
    # Prepare output file path
    if output_file_path is None:
        base_name = Path(msg_file_path).stem
        output_file_path = f"{base_name}.md"
    
    # Extract email content
    markdown_content = []
    
    # Header information
    markdown_content.append("# Email Message\n\n")
    
    if msg.subject:
        subject = clean_text(msg.subject)
        markdown_content.append(f"**Subject:** {subject}\n\n")
    
    if msg.sender:
        sender = clean_text(msg.sender)
        markdown_content.append(f"**From:** {sender}\n\n")
    
    if msg.to:
        to = clean_text(msg.to)
        markdown_content.append(f"**To:** {to}\n\n")
    
    if msg.cc:
        cc = clean_text(msg.cc)
        markdown_content.append(f"**CC:** {cc}\n\n")
    
    if msg.date:
        markdown_content.append(f"**Date:** {msg.date}\n\n")
    
    markdown_content.append("---\n\n")
    
    # Try multiple ways to get the email body
    body_text = None
    
    # Method 1: Try msg.body
    if hasattr(msg, 'body') and msg.body:
        body_text = clean_text(msg.body)
    
    # Method 2: Try msg.htmlBody and strip HTML if body is empty
    if not body_text and hasattr(msg, 'htmlBody') and msg.htmlBody:
        html_body = clean_text(msg.htmlBody)
        if html_body:
            # More comprehensive HTML/VML cleaning
            import re
            # Remove VML styles and CSS
            body_text = re.sub(r'v\\:[^{]*{[^}]*}', '', html_body)
            body_text = re.sub(r'o\\:[^{]*{[^}]*}', '', body_text)
            body_text = re.sub(r'w\\:[^{]*{[^}]*}', '', body_text)
            body_text = re.sub(r'\.shape[^{]*{[^}]*}', '', body_text)
            
            # Remove HTML tags
            body_text = re.sub(r'<[^>]+>', '', body_text)
            
            # Clean up HTML entities
            body_text = re.sub(r'&nbsp;', ' ', body_text)
            body_text = re.sub(r'&[a-zA-Z0-9#]+;', '', body_text)
            
            # Remove multiple spaces and clean up
            body_text = re.sub(r'\s+', ' ', body_text)
            body_text = clean_text(body_text)
    
    # Method 3: Try msg.rtfBody if available
    if not body_text and hasattr(msg, 'rtfBody') and msg.rtfBody:
        rtf_body = clean_text(str(msg.rtfBody))
        if rtf_body:
            body_text = rtf_body
    
    # Method 4: Debug - print available attributes
    if not body_text:
        print("Debug: Available message attributes:")
        for attr in dir(msg):
            if not attr.startswith('_') and 'body' in attr.lower():
                try:
                    value = getattr(msg, attr)
                    if value:
                        print(f"  {attr}: {type(value)} - {len(str(value)) if value else 0} chars")
                except:
                    print(f"  {attr}: Error accessing")
    
    # Email body
    if body_text and body_text.strip():
        markdown_content.append("## Message Body\n\n")
        # Clean up the body text and convert to markdown-friendly format
        paragraphs = [p.strip() for p in body_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in body_text.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            if paragraph:
                markdown_content.append(f"{paragraph}\n\n")
    else:
        markdown_content.append("## Message Body\n\n*[No readable message body found]*\n\n")
    
    # Attachments
    if msg.attachments:
        markdown_content.append("## Attachments\n\n")
        attachment_dir = None
        
        for i, attachment in enumerate(msg.attachments, 1):
            attachment_name = attachment.longFilename or attachment.shortFilename or f"attachment_{i}"
            attachment_name = clean_text(attachment_name)
            
            # Save embedded images
            if attachment_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                try:
                    if not attachment_dir:
                        # Create attachments directory
                        base_name = Path(output_file_path).stem
                        attachment_dir = f"{base_name}_attachments"
                        os.makedirs(attachment_dir, exist_ok=True)
                    
                    attachment_path = os.path.join(attachment_dir, attachment_name)
                    # Extract attachment data and save
                    with open(attachment_path, 'wb') as f:
                        f.write(attachment.data)
                    
                    # Add image reference in markdown with proper path encoding
                    # URL encode spaces and special characters for markdown
                    import urllib.parse
                    encoded_path = urllib.parse.quote(attachment_path)
                    markdown_content.append(f"- ![{attachment_name}]({encoded_path})\n")
                    print(f"Saved image: {attachment_path}")
                except Exception as e:
                    print(f"Could not save attachment {attachment_name}: {e}")
                    markdown_content.append(f"- {attachment_name} *(could not extract)*\n")
            else:
                markdown_content.append(f"- {attachment_name}\n")
    
    # Write to output file
    try:
        with open(output_file_path, 'w', encoding='utf-8', errors='replace') as f:
            f.writelines(markdown_content)
        print(f"Successfully converted '{msg_file_path}' to '{output_file_path}'")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Microsoft Outlook .msg files to Markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python msg_to_md.py email.msg
  python msg_to_md.py email.msg output.md
  python msg_to_md.py "seed_files/RE_ Fin POC Fonds distincts.msg"
        """
    )
    
    parser.add_argument('input_file', help='Input .msg file path')
    parser.add_argument('output_file', nargs='?', help='Output .md file path (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Check if input file has .msg extension
    if not args.input_file.lower().endswith('.msg'):
        print(f"Warning: Input file doesn't have .msg extension")
    
    # Convert the file
    success = msg_to_markdown(args.input_file, args.output_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
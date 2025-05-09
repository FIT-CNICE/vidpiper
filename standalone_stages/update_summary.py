#!/usr/bin/env python3
"""
Standalone script for updating an existing summary based on human feedback.

This script takes a summary file and feedback text, then creates an updated
summary incorporating the feedback.

Usage:
    python update_summary.py --summary_file path/to/summary.md --feedback "Your feedback text" [--output_dir path/to/output]
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidpiper.stages import create_summary_updater


def update_summary(summary_file, feedback, output_dir=None, provider="gemini"):
    """
    Update a summary based on feedback.

    Args:
        summary_file (str): Path to the summary file to update
        feedback (str): Human feedback text for improving the summary
        output_dir (str, optional): Directory to save the updated summary
        provider (str, optional): Preferred LLM provider (gemini, anthropic, or openai)
    """
    # Verify that summary_file exists
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)

    # Validate feedback
    if not feedback or not feedback.strip():
        print("Error: Feedback text is empty")
        sys.exit(1)

    # Set output directory if not provided
    if not output_dir:
        output_dir = os.path.dirname(summary_file)

    print("Updating summary based on feedback...")
    print(f"Summary file: {summary_file}")
    print(f"Output directory: {output_dir}")
    print(f"Preferred provider: {provider}")

    # Create summary updater
    updater = create_summary_updater(
        max_tokens=2000, output_dir=output_dir, preferred_provider=provider
    )

    # Update the summary
    try:
        updated_summary_path = updater.update_summary_file(
            summary_file=summary_file, feedback=feedback, output_dir=output_dir
        )
        print(f"\nSuccess! Updated summary saved to: {updated_summary_path}")
    except Exception as e:
        print(f"Error updating summary: {e}")
        sys.exit(1)
    finally:
        updater.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update a summary based on human feedback"
    )
    parser.add_argument(
        "--summary_file",
        required=True,
        help="Path to the summary file to update",
    )
    parser.add_argument(
        "--feedback",
        required=True,
        help="Human feedback text for improving the summary",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the updated summary (defaults to same as summary file)",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "anthropic", "openai"],
        help="Preferred LLM provider",
    )

    args = parser.parse_args()

    update_summary(
        summary_file=args.summary_file,
        feedback=args.feedback,
        output_dir=args.output_dir,
        provider=args.provider,
    )

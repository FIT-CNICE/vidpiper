#!/usr/bin/env python3
"""
Format summaries into Marp slide decks.

This script processes markdown summary files (with _sum.md suffix) and formats them
into Marp slide decks using LLMs. It can process a single file or all files in a directory.

Note: This standalone script provides the core formatting functionality that is also
integrated into the main summerizer_cli.py script. You can use this script directly
for formatting operations, or use the main CLI with the "--run-mode format" option.

Examples:
    # Format a single file
    python format_summaries.py path/to/summary_sum.md -o output/dir

    # Format all summary files in a directory
    python format_summaries.py path/to/summaries/dir -o output/dir
"""

import os
import argparse
import sys
from typing import Optional

# NOTE: This script provides the same functionality as running:
# python summerizer_cli.py --run-mode format

from vidpiper.stages import create_summary_formatter
from vidpiper.core.data_classes import PipelineResult


def process_file(file_path: str, output_dir: Optional[str] = None) -> str:
    """Process a single summary file and return the path to the formatted file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    # If output_dir is not specified, use the same directory as the input file
    if not output_dir:
        output_dir = os.path.dirname(file_path)

    # Create the formatter with the specified output directory
    formatter = create_summary_formatter(output_dir=output_dir)

    # Create a dummy pipeline result
    data = PipelineResult(
        video_path="",  # Not needed for formatting
        summary_file=file_path,
        output_dir=output_dir,
    )

    # Run the formatter
    result = formatter.run(data)

    # Return the path to the formatted file
    return result.formatted_file


def process_directory(directory: str, output_dir: Optional[str] = None) -> list:
    """Process all summary files in a directory and return the paths to the formatted files."""
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return []

    # If output_dir is not specified, use the input directory
    if not output_dir:
        output_dir = directory

    # Create the formatter with the specified output directory
    formatter = create_summary_formatter(output_dir=output_dir)

    # Use the formatter's standalone method to process the directory
    formatted_files = formatter.format_directory(directory)

    return formatted_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Format summaries into Marp slide decks."
    )
    parser.add_argument(
        "path",
        help="Path to a summary file or directory containing summary files",
    )
    parser.add_argument(
        "-o", "--output-dir", help="Directory to save formatted files"
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="gemini",
        choices=["gemini", "anthropic", "openai"],
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-2.0-flash",
        help="Model to use (default: gemini-2.0-flash)",
    )

    args = parser.parse_args()

    # Check if the path exists
    if not os.path.exists(args.path):
        print(f"Error: Path not found: {args.path}")
        return 1

    # Process the path based on whether it's a file or directory
    if os.path.isfile(args.path):
        # Check if the file has the _sum.md suffix
        if not args.path.endswith("_sum.md"):
            print(f"Warning: File does not have _sum.md suffix: {args.path}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != "y":
                return 0

        # Process the file
        formatted_file = process_file(args.path, args.output_dir)
        if formatted_file:
            print(f"Formatted file saved to: {formatted_file}")
        else:
            print("Failed to format file.")
            return 1
    else:
        # Process the directory
        formatted_files = process_directory(args.path, args.output_dir)
        if formatted_files:
            print(f"Formatted {len(formatted_files)} files:")
            for file in formatted_files:
                print(f"  - {file}")
        else:
            print("No summary files found or failed to format files.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Summary generation script for processed video scenes.

This standalone script generates a markdown summary from previously
processed scenes with screenshots and transcripts.
"""

import os
import sys
import json
import argparse

# Add the project root directory to sys.path to access the vidpiper module
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import vidpiper modules after path setup
from vidpiper.core import PipelineResult  # noqa: E402
from vidpiper.stages import create_summary_generator  # noqa: E402

# NOTE: This script provides the same functionality as running:
# python vidpiper_cli.py --run-mode summarize


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a summary from processed scenes"
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSON file with processed scenes",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output (defaults to directory of input file)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model name (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Maximum tokens per API response (default: 1500)",
    )
    parser.add_argument(
        "--llm-provider",
        default="gemini",
        choices=["anthropic", "openai", "gemini"],
        help="Preferred LLM provider (default: gemini)",
    )

    return parser.parse_args()


def get_output_dir(args):
    """Determine the output directory based on arguments or input directory."""
    if args.output_dir:
        return args.output_dir

    # Use the directory of the input file
    return os.path.dirname(os.path.abspath(args.input_file))


def main():
    """Generate summary from processed scenes."""
    args = parse_args()

    # Ensure input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Load input data
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Create the summary generator
    generator = create_summary_generator(
        model=args.model,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        preferred_provider=args.llm_provider,
    )

    # Run summary generation
    try:
        print(f"Generating summary for {len(initial_data.scenes)} scenes...")
        result = generator.run(initial_data)

        print("Summary generation complete.")

        if result.summary_file:
            print(f"Summary saved to: {result.summary_file}")

    except Exception as e:
        print(f"Summary generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

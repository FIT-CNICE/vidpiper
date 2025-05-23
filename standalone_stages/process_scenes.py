#!/usr/bin/env python3
"""
Scene processing script for videos.

This standalone script processes previously detected scenes by extracting
screenshots and transcripts, saving the results for summary generation.
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
from vidpiper.stages import create_scene_processor  # noqa: E402

# NOTE: This script provides the same functionality as running:
# python vidpiper_cli.py --run-mode process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process scenes with screenshots and transcripts"
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSON file with detected scenes",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output (defaults to directory of input file)",
    )
    parser.add_argument(
        "--output-file",
        default="processed_scenes.json",
        help="Filename to save processing results (processed_scenes.json)",
    )
    parser.add_argument(
        "--use-whisper",
        action="store_true",
        default=False,
        help="Use Whisper for transcription if available (default: False)",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (small)",
    )

    return parser.parse_args()


def get_output_dir(args):
    """Determine the output directory based on arguments or input directory."""
    if args.output_dir:
        return args.output_dir

    # Use the directory of the input file
    return os.path.dirname(os.path.abspath(args.input_file))


def main():
    """Process scenes in a video file."""
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

    # Create the scene processor
    processor = create_scene_processor(
        output_dir=output_dir,
        use_whisper=args.use_whisper,
        whisper_model=args.whisper_model,
    )

    # Run processing
    try:
        print(f"Processing {len(initial_data.scenes)} scenes...")
        result = processor.run(initial_data)

        print(
            f"Scene processing complete. Processed {len(result.scenes)} scenes."
        )

        # Save result to the output file
        output_file = os.path.join(output_dir, args.output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved processed scenes to: {output_file}")

    except Exception as e:
        print(f"Scene processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

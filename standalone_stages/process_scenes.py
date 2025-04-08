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

from video_summarizer.core import PipelineResult
from video_summarizer.stages import create_scene_processor

# NOTE: This script provides the same functionality as running:
# python summerizer_cli.py --run-mode process


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
        help="Filename to save processing results (default: processed_scenes.json)",
    )
    parser.add_argument(
        "--use-whisper",
        action="store_true",
        help="Use Whisper for transcription if available (default: False)",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: small)",
    )

    return parser.parse_args()


def get_output_dir(args):
    """Determine the output directory based on arguments or input file directory."""
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

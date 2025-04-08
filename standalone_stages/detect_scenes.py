#!/usr/bin/env python3
"""
Scene detection script for videos.

This standalone script detects scenes in a video file and saves the results
to a JSON file that can be used by other pipeline stages.
"""

import os
import sys
import json
import argparse

from vidpiper.core import PipelineResult
from vidpiper.stages import create_scene_detector

# NOTE: This script provides the same functionality as running:
# python vidpiper_cli.py <video_path> --run-mode detect


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect scenes in a video file"
    )

    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output (defaults to <video_name>_output)",
    )
    parser.add_argument(
        "--output-file",
        default="detected_scenes.json",
        help="Filename to save detection results (default: detected_scenes.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=35.0,
        help="Threshold for scene detection (default: 35.0)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=64,
        help="Downscale factor for scene detection (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds for scene detection (default: 180)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=20.0,
        help="Maximum size of scenes in MB (default: 20.0)",
    )
    parser.add_argument(
        "--skip-start",
        type=float,
        default=60.0,
        help="Number of seconds to skip at the beginning of the video (default: 60.0)",
    )
    parser.add_argument(
        "--skip-end",
        type=float,
        default=0.0,
        help="Number of seconds to skip at the end of the video (default: 0.0)",
    )
    parser.add_argument(
        "--max-scene",
        type=int,
        default=None,
        help="Maximum number of scenes to detect (default: auto-calculated based on duration)",
    )

    return parser.parse_args()


def get_output_dir(args):
    """Determine the output directory based on arguments or video filename."""
    if args.output_dir:
        return args.output_dir

    # Create output directory based on video filename
    video_basename = os.path.basename(args.video_path)
    video_name = os.path.splitext(video_basename)[0]
    return f"{video_name}_output"


def main():
    """Run scene detection on a video file."""
    args = parse_args()

    # Ensure video file exists
    if not os.path.exists(args.video_path):
        print(f"Video file not found: {args.video_path}")
        sys.exit(1)

    # Create output directory
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Create the scene detector
    detector = create_scene_detector(
        threshold=args.threshold,
        downscale_factor=args.downscale,
        timeout_seconds=args.timeout,
        max_size_mb=args.max_size,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        max_scene=args.max_scene,
    )

    # Create initial pipeline data
    initial_data = PipelineResult(video_path=args.video_path)

    # Run detection
    try:
        print(f"Detecting scenes in {args.video_path}...")
        result = detector.run(initial_data)

        print(
            f"Scene detection complete. Detected {len(result.scenes)} scenes."
        )

        # Save result to the output file
        output_file = os.path.join(output_dir, args.output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved scene detection results to: {output_file}")

    except Exception as e:
        print(f"Scene detection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

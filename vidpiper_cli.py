#!/usr/bin/env python3
"""
VidPiper CLI

This script provides a command-line interface for running the vidpiper
pipeline stages. It supports executing stages as individual commands.
"""

import os
import sys
import json
import argparse

from vidpiper.core import Pipeline, PipelineResult
from vidpiper.stages import (
    create_scene_detector,
    create_scene_processor,
    create_summary_generator,
    create_summary_formatter,
    create_summary_updater,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VidPiper: AI agent for summarizing webinar videos",
    )

    # Add subparsers
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Pipeline stage to run"
    )

    # 1. Scene Detection
    detect_parser = subparsers.add_parser(
        "detect", help="Detect scenes in a video"
    )
    detect_parser.add_argument("video_path", help="Path to the video file")
    detect_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output (defaults to <video_name>_output)",
    )
    detect_parser.add_argument(
        "--output-file",
        default="detected_scenes.json",
        help="Filename to save detection results (default: detected_scenes.json)",
    )
    detect_parser.add_argument(
        "--threshold",
        type=float,
        default=35.0,
        help="Threshold for scene detection (default: 35.0)",
    )
    detect_parser.add_argument(
        "--downscale",
        type=int,
        default=64,
        help="Downscale factor for scene detection (default: 64)",
    )
    detect_parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds for scene detection (default: 180)",
    )
    detect_parser.add_argument(
        "--max-size",
        type=float,
        default=20.0,
        help="Maximum size of scenes in MB (default: 20.0)",
    )
    detect_parser.add_argument(
        "--skip-start",
        type=float,
        default=60.0,
        help="Number of seconds to skip at the beginning of the video (default: 60.0)",
    )
    detect_parser.add_argument(
        "--skip-end",
        type=float,
        default=0.0,
        help="Number of seconds to skip at the end of the video (default: 0.0)",
    )
    detect_parser.add_argument(
        "--max-scene",
        type=int,
        default=None,
        help="Maximum number of scenes to detect (default: auto-calculated based on duration)",
    )
    detect_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save/load pipeline checkpoints",
    )

    # 2. Scene Processing
    process_parser = subparsers.add_parser(
        "process", help="Process scenes with screenshots and transcripts"
    )
    process_parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSON file with detected scenes",
    )
    process_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output (defaults to directory of input file)",
    )
    process_parser.add_argument(
        "--output-file",
        default="processed_scenes.json",
        help="Filename to save processing results (processed_scenes.json)",
    )
    process_parser.add_argument(
        "--use-whisper",
        action="store_true",
        default=False,
        help="Use Whisper for transcription if available (default: False)",
    )
    process_parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (small)",
    )
    process_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save/load pipeline checkpoints",
    )
    process_parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Specific checkpoint file to load",
    )

    # 3. Summary Generation
    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate summaries from processed scenes"
    )
    summarize_parser.add_argument(
        "--input-file",
        required=True,
        help="Input JSON file with processed scenes",
    )
    summarize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output (defaults to directory of input file)",
    )
    summarize_parser.add_argument(
        "--output-file",
        default=None,
        help="Filename for the output summary (defaults to auto-generated name)",
    )
    summarize_parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model name (default: gemini-2.0-flash)",
    )
    summarize_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Maximum tokens per API response (default: 1500)",
    )
    summarize_parser.add_argument(
        "--llm-provider",
        default="gemini",
        choices=["anthropic", "openai", "gemini"],
        help="Preferred LLM provider (default: gemini)",
    )
    summarize_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save/load pipeline checkpoints",
    )
    summarize_parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Specific checkpoint file to load",
    )

    # 4. Summary Formatting
    format_parser = subparsers.add_parser(
        "format", help="Format summaries into presentation slides"
    )
    format_parser.add_argument(
        "path",
        help="Path to a summary file or directory containing summary files",
        nargs="?",
        default=None,
    )
    format_parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save formatted files",
    )
    format_parser.add_argument(
        "-p", "--provider",
        default="gemini",
        choices=["gemini", "anthropic", "openai"],
        help="LLM provider to use (default: gemini)",
        dest="llm_provider",
    )
    format_parser.add_argument(
        "-m", "--model",
        default="gemini-2.0-flash",
        help="Model to use (default: gemini-2.0-flash)",
    )
    # Keep legacy options for backward compatibility
    format_parser.add_argument(
        "--format-single",
        default=None,
        help="Path to a single summary file to format (legacy option)",
    )
    format_parser.add_argument(
        "--format-dir",
        default=None,
        help="Directory containing summary files to format (legacy option)",
    )
    format_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save/load pipeline checkpoints",
    )
    format_parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Specific checkpoint file to load",
    )
    format_parser.add_argument(
        "--video-path",
        default=None,
        help="Optional path to the original video file",
    )

    # 5. Summary Update
    update_parser = subparsers.add_parser(
        "update", help="Update existing summaries"
    )
    update_parser.add_argument(
        "--summary_file",
        required=True,
        help="Path to the summary file to update",
    )
    update_parser.add_argument(
        "--feedback",
        default=None,
        help="Human feedback text for improving the summary",
    )
    update_parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save the updated summary (defaults to same as summary file)",
        dest="output_dir",
    )
    update_parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "anthropic", "openai"],
        help="Preferred LLM provider",
        dest="llm_provider",
    )
    update_parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="LLM model name (default: gemini-2.0-flash)",
    )
    update_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="Maximum tokens per API response (default: 1500)",
    )
    # Add legacy option for backward compatibility
    update_parser.add_argument(
        "--summary-file",
        dest="summary_file_alt",
    )
    update_parser.add_argument(
        "--update-prompt",
        dest="feedback_alt",
        help=argparse.SUPPRESS,  # Hidden parameter
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


def run_full_pipeline(args):
    """Run the complete video summarization pipeline."""
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Create the pipeline
    pipeline = Pipeline()

    # Set checkpoint dir if provided
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        pipeline.set_checkpoint_dir(args.checkpoint_dir)

    # Add pipeline stages
    pipeline.add_stage(
        create_scene_detector(
            threshold=args.threshold,
            downscale_factor=args.downscale,
            timeout_seconds=args.timeout,
            max_size_mb=args.max_size,
            skip_start=args.skip_start,
            skip_end=args.skip_end,
            max_scene=args.max_scene,
        )
    )

    pipeline.add_stage(
        create_scene_processor(
            output_dir=output_dir,
            use_whisper=args.use_whisper,
            whisper_model=args.whisper_model,
        )
    )

    pipeline.add_stage(
        create_summary_generator(
            model=args.model,
            max_tokens=args.max_tokens,
            output_dir=output_dir,
            preferred_provider=args.llm_provider,
        )
    )

    # Add the formatter stage
    pipeline.add_stage(
        create_summary_formatter(
            model=args.model,
            max_tokens=args.max_tokens,
            output_dir=output_dir,
            preferred_provider=args.llm_provider,
        )
    )

    # Create initial pipeline data
    initial_data = PipelineResult(video_path=args.video_path)

    # Run the pipeline
    try:
        result = pipeline.run(initial_data)
        print(f"Pipeline complete. Output saved to: {output_dir}")

        if result.summary_file:
            print(f"Markdown summary: {result.summary_file}")

        if result.formatted_file:
            print(f"Formatted Marp deck: {result.formatted_file}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


def run_scene_detection(args):
    """Run only the scene detection stage."""
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create the pipeline with only the scene detector
    pipeline = Pipeline()

    if args.checkpoint_dir:
        pipeline.set_checkpoint_dir(args.checkpoint_dir)

    detector = create_scene_detector(
        threshold=args.threshold,
        downscale_factor=args.downscale,
        timeout_seconds=args.timeout,
        max_size_mb=args.max_size,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        max_scene=args.max_scene,
    )

    pipeline.add_stage(detector)

    # Create initial pipeline data
    initial_data = PipelineResult(video_path=args.video_path)

    # Run detection stage
    try:
        result = pipeline.run(initial_data)
        print(
            f"Scene detection complete. Detected {len(result.scenes)} scenes."
        )

        # Save result to a file for later use
        output_file = os.path.join(output_dir, "detected_scenes.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved scene detection results to: {output_file}")

    except Exception as e:
        print(f"Scene detection failed: {e}")
        sys.exit(1)


def run_scene_processing(args):
    """Run only the scene processing stage."""
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create the pipeline with only the scene processor
    pipeline = Pipeline()

    if args.checkpoint_dir:
        pipeline.set_checkpoint_dir(args.checkpoint_dir)

    processor = create_scene_processor(
        output_dir=output_dir,
        use_whisper=args.use_whisper,
        whisper_model=args.whisper_model,
    )

    pipeline.add_stage(processor)

    # Get initial data from checkpoint or detected_scenes.json
    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        # Load from specified checkpoint
        with open(args.checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)
    else:
        # Try to find detected_scenes.json
        scenes_file = os.path.join(output_dir, "detected_scenes.json")
        if not os.path.exists(scenes_file):
            print(f"Cannot find detected scenes file: {scenes_file}")
            print(
                "Please run scene detection first or specify a checkpoint file."
            )
            sys.exit(1)

        with open(scenes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)

    # Run processing stage
    try:
        result = pipeline.run(initial_data)
        print(
            f"Scene processing complete. Processed {len(result.scenes)} scenes."
        )

        # Save result to a file for later use
        output_file = os.path.join(output_dir, "processed_scenes.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved processed scenes to: {output_file}")

    except Exception as e:
        print(f"Scene processing failed: {e}")
        sys.exit(1)


def run_summary_generation(args):
    """Run only the summary generation stage."""
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create the pipeline with only the summary generator
    pipeline = Pipeline()

    if args.checkpoint_dir:
        pipeline.set_checkpoint_dir(args.checkpoint_dir)

    generator = create_summary_generator(
        model=args.model,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        preferred_provider=args.llm_provider,
    )

    pipeline.add_stage(generator)

    # Get initial data from checkpoint or processed_scenes.json
    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        # Load from specified checkpoint
        with open(args.checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)
    else:
        # Try to find processed_scenes.json
        scenes_file = os.path.join(output_dir, "processed_scenes.json")
        if not os.path.exists(scenes_file):
            print(f"Cannot find processed scenes file: {scenes_file}")
            print(
                "Please run scene processing first or specify a checkpoint file."
            )
            sys.exit(1)

        with open(scenes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)

    # Run summary generation stage
    try:
        result = pipeline.run(initial_data)
        print("Summary generation complete.")

        if result.summary_file:
            print(f"Markdown summary saved to: {result.summary_file}")

    except Exception as e:
        print(f"Summary generation failed: {e}")
        sys.exit(1)


def run_summary_formatting(args):
    """Run only the summary formatting stage."""
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create the pipeline with only the summary formatter
    pipeline = Pipeline()

    if args.checkpoint_dir:
        pipeline.set_checkpoint_dir(args.checkpoint_dir)

    formatter = create_summary_formatter(
        model=args.model,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        preferred_provider=args.llm_provider,
    )

    pipeline.add_stage(formatter)

    # Determine what to format
    if args.format_single and os.path.exists(args.format_single):
        # Format a single file
        print(f"Formatting single file: {args.format_single}")
        initial_data = PipelineResult(
            video_path=args.video_path,
            summary_file=args.format_single,
            output_dir=output_dir,
        )
    elif args.format_dir and os.path.isdir(args.format_dir):
        # Format all summary files in a directory
        print(f"Formatting all summary files in directory: {args.format_dir}")
        # Initialize with dummy data - the formatter will search the directory
        initial_data = PipelineResult(
            video_path=args.format_dir,  # Use directory as video_path for search
            output_dir=output_dir,
        )
    elif args.checkpoint_file and os.path.exists(args.checkpoint_file):
        # Load from specified checkpoint
        with open(args.checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)
    else:
        # Try to find summary file from previous stage
        summary_pattern = os.path.join(output_dir, "*_sum.md")
        import glob

        summary_files = glob.glob(summary_pattern)

        if not summary_files:
            print(f"Cannot find summary files in: {output_dir}")
            print(
                "Please run summary generation first or specify a file/directory to format."
            )
            sys.exit(1)

        # Use the most recent summary file
        latest_summary = max(summary_files, key=os.path.getmtime)
        print(f"Using most recent summary file: {latest_summary}")

        initial_data = PipelineResult(
            video_path=args.video_path,
            summary_file=latest_summary,
            output_dir=output_dir,
        )

    # Run formatting stage
    try:
        result = pipeline.run(initial_data)
        print("Summary formatting complete.")

        if result.formatted_file:
            print(f"Formatted Marp deck saved to: {result.formatted_file}")
        elif "formatted_summaries" in result.metadata:
            formatted_files = result.metadata["formatted_summaries"]
            print(f"Formatted {len(formatted_files)} summary files:")
            for f in formatted_files:
                print(f"  - {f}")

    except Exception as e:
        print(f"Summary formatting failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Handle the different commands
    if args.command == "detect":
        # Ensure video file exists
        if not os.path.exists(args.video_path):
            print(f"Error: Video file not found: {args.video_path}")
            sys.exit(1)

        # Determine output directory
        output_dir = args.output_dir
        if not output_dir:
            video_basename = os.path.basename(args.video_path)
            video_name = os.path.splitext(video_basename)[0]
            output_dir = f"{video_name}_output"

        os.makedirs(output_dir, exist_ok=True)

        # Create and run scene detector
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

            print(f"Scene detection complete. Detected {len(result.scenes)} scenes.")

            # Save result to the output file
            output_file = os.path.join(output_dir, args.output_file)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

            print(f"Saved scene detection results to: {output_file}")

        except Exception as e:
            print(f"Error: Scene detection failed: {e}")
            sys.exit(1)

    elif args.command == "process":
        # Ensure input file exists
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            sys.exit(1)

        # Determine output directory
        output_dir = args.output_dir
        if not output_dir:
            # Use the directory of the input file
            output_dir = os.path.dirname(os.path.abspath(args.input_file))

        os.makedirs(output_dir, exist_ok=True)

        # Load input data
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            initial_data = PipelineResult.from_dict(data)
        except Exception as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)

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

            print(f"Scene processing complete. Processed {len(result.scenes)} scenes.")

            # Save result to the output file
            output_file = os.path.join(output_dir, args.output_file)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

            print(f"Saved processed scenes to: {output_file}")

        except Exception as e:
            print(f"Error: Scene processing failed: {e}")
            sys.exit(1)

    elif args.command == "summarize":
        # Ensure input file exists
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            sys.exit(1)

        # Determine output directory
        output_dir = args.output_dir
        if not output_dir:
            # Use the directory of the input file
            output_dir = os.path.dirname(os.path.abspath(args.input_file))

        os.makedirs(output_dir, exist_ok=True)

        # Load input data
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            initial_data = PipelineResult.from_dict(data)
        except Exception as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)

        # Create the summary generator with optional output filename
        generator = create_summary_generator(
            model=args.model,
            max_tokens=args.max_tokens,
            output_dir=output_dir,
            preferred_provider=args.llm_provider,
            # output_filename=args.output_file,
        )

        # Run summary generation
        try:
            print("Generating summary...")
            result = generator.run(initial_data)

            print("Summary generation complete.")

            if result.summary_file:
                print(f"Summary saved to: {result.summary_file}")
            else:
                print("No summary file was produced.")

        except Exception as e:
            print(f"Error: Summary generation failed: {e}")
            sys.exit(1)

    elif args.command == "format":
        # Handle different input methods: positional path arg, format-single, or format-dir
        path = args.path or args.format_single or args.format_dir

        if not path:
            print("Error: You must provide a path to a summary file or directory.")
            print("Use either a positional argument or --format-single/--format-dir options.")
            sys.exit(1)

        # Check if the path exists
        if not os.path.exists(path):
            print(f"Error: Path not found: {path}")
            sys.exit(1)

        # Determine output directory
        output_dir = args.output_dir
        if not output_dir:
            if os.path.isfile(path):
                output_dir = os.path.dirname(os.path.abspath(path))
            else:
                output_dir = path

        os.makedirs(output_dir, exist_ok=True)

        # Create the summary formatter
        formatter = create_summary_formatter(
            model=args.model,
            max_tokens=args.max_tokens,
            output_dir=output_dir,
            preferred_provider=args.llm_provider,
        )

        # Process based on path type
        if os.path.isfile(path):
            # Check if the file has the _sum.md suffix
            if not path.endswith("_sum.md"):
                print(f"Warning: File does not have _sum.md suffix: {path}")
                print("It may not be a summary file. Continuing anyway...")

            # Process file
            print(f"Formatting single file: {path}")
            initial_data = PipelineResult(
                video_path=args.video_path,
                summary_file=path,
                output_dir=output_dir,
            )

            # Run formatter
            try:
                result = formatter.run(initial_data)
                print("Summary formatting complete.")

                if result.formatted_file:
                    print(f"Formatted Marp deck saved to: {result.formatted_file}")
                else:
                    print("No formatted file was produced.")

            except Exception as e:
                print(f"Error: Summary formatting failed: {e}")
                sys.exit(1)
        else:
            # Process directory
            print(f"Formatting all summary files in directory: {path}")

            try:
                # Use formatter's directory processing capabilities
                formatted_files = formatter.format_directory(path)

                if formatted_files:
                    print(f"Formatted {len(formatted_files)} files:")
                    for f in formatted_files:
                        print(f"  - {f}")
                else:
                    print("No summary files found or failed to format files.")

            except Exception as e:
                print(f"Error: Summary formatting failed: {e}")
                sys.exit(1)

    elif args.command == "update":
        # Handle both naming conventions for backward compatibility
        summary_file = args.summary_file or args.summary_file_alt
        feedback = args.feedback or args.feedback_alt

        # Ensure summary file exists
        if not os.path.exists(summary_file):
            print(f"Error: Summary file not found: {summary_file}")
            sys.exit(1)

        # Validate feedback
        if not feedback or not feedback.strip():
            print("Error: Feedback text is empty.")
            print("Please provide feedback using the --feedback parameter.")
            sys.exit(1)

        # Determine output directory
        output_dir = args.output_dir
        if not output_dir:
            # Use the directory of the summary file
            output_dir = os.path.dirname(os.path.abspath(summary_file))

        os.makedirs(output_dir, exist_ok=True)

        # Create the summary updater
        updater = create_summary_updater(
            model=args.model,
            max_tokens=args.max_tokens,
            output_dir=output_dir,
            preferred_provider=args.llm_provider,
        )

        # Run update
        try:
            print(f"Updating summary: {summary_file}")
            print(f"Based on feedback: {feedback}")

            # Use the direct update method
            updated_summary_path = updater.update_summary_file(
                summary_file=summary_file,
                feedback=feedback,
                output_dir=output_dir
            )

            print("Summary update complete.")
            print(f"Updated summary saved to: {updated_summary_path}")

        except Exception as e:
            print(f"Error: Summary update failed: {e}")
            sys.exit(1)
        finally:
            updater.cleanup()


if __name__ == "__main__":
    main()

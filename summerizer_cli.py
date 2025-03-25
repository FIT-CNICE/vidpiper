#!/usr/bin/env python3
"""
Video Summarizer CLI

This script provides a command-line interface for summarizing videos
using the video_summarizer package. It allows for running the entire pipeline
or individual stages as needed.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

from video_summarizer.core import Pipeline, PipelineResult
from video_summarizer.stages import (
    create_scene_detector,
    create_scene_processor,
    create_summary_generator
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown summary of a video with screenshots")
    
    # Main options
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save output (defaults to <video_name>_output)")
    parser.add_argument(
        "--run-mode", default="full",
        choices=["full", "detect", "process", "summarize"],
        help="Mode to run: full pipeline or individual stages (default: full)")
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory to save/load pipeline checkpoints")
    parser.add_argument(
        "--checkpoint-file", default=None,
        help="Specific checkpoint file to load (for running a specific stage)")
    
    # Scene detection options
    detect_group = parser.add_argument_group('Scene Detection Options')
    detect_group.add_argument(
        "--threshold", type=float, default=35.0,
        help="Threshold for scene detection (default: 35.0)")
    detect_group.add_argument(
        "--downscale", type=int, default=64,
        help="Downscale factor for scene detection (default: 64)")
    detect_group.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout in seconds for scene detection (default: 180)")
    detect_group.add_argument(
        "--max-size", type=float, default=20.0,
        help="Maximum size of scenes in MB (default: 20.0)")
    detect_group.add_argument(
        "--skip-start", type=float, default=60.0,
        help="Number of seconds to skip at the beginning of the video (default: 60.0)")
    detect_group.add_argument(
        "--skip-end", type=float, default=0.0,
        help="Number of seconds to skip at the end of the video (default: 0.0)")
    detect_group.add_argument(
        "--max-scene", type=int, default=None,
        help="Maximum number of scenes to detect (default: auto-calculated based on duration)")
    
    # Scene processing options
    process_group = parser.add_argument_group('Scene Processing Options')
    process_group.add_argument(
        "--use-whisper", action="store_true",
        help="Use Whisper for transcription if available (default: False)")
    process_group.add_argument(
        "--whisper-model", default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: small)")
    
    # Summary generation options
    summary_group = parser.add_argument_group('Summary Generation Options')
    summary_group.add_argument(
        "--model", default="claude-3-7-sonnet-20250219",
        help="LLM model name (default: claude-3-7-sonnet-20250219)")
    summary_group.add_argument(
        "--max-tokens", type=int, default=1500,
        help="Maximum tokens per API response (default: 1500)")
    summary_group.add_argument(
        "--llm-provider", default="gemini",
        choices=["anthropic", "openai", "gemini"],
        help="Preferred LLM provider (default: gemini)")
    
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
    pipeline.add_stage(create_scene_detector(
        threshold=args.threshold,
        downscale_factor=args.downscale,
        timeout_seconds=args.timeout,
        max_size_mb=args.max_size,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        max_scene=args.max_scene
    ))
    
    pipeline.add_stage(create_scene_processor(
        output_dir=output_dir,
        use_whisper=args.use_whisper,
        whisper_model=args.whisper_model
    ))
    
    pipeline.add_stage(create_summary_generator(
        model=args.model,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
        preferred_provider=args.llm_provider
    ))
    
    # Create initial pipeline data
    initial_data = PipelineResult(video_path=args.video_path)
    
    # Run the pipeline
    try:
        result = pipeline.run(initial_data)
        print(f"Pipeline complete. Output saved to: {output_dir}")
        
        if result.summary_file:
            print(f"Markdown summary: {result.summary_file}")
            
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
        max_scene=args.max_scene
    )
    
    pipeline.add_stage(detector)
    
    # Create initial pipeline data
    initial_data = PipelineResult(video_path=args.video_path)
    
    # Run detection stage
    try:
        result = pipeline.run(initial_data)
        print(f"Scene detection complete. Detected {len(result.scenes)} scenes.")
        
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
        whisper_model=args.whisper_model
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
            print("Please run scene detection first or specify a checkpoint file.")
            sys.exit(1)
        
        with open(scenes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)
    
    # Run processing stage
    try:
        result = pipeline.run(initial_data)
        print(f"Scene processing complete. Processed {len(result.scenes)} scenes.")
        
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
        preferred_provider=args.llm_provider
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
            print("Please run scene processing first or specify a checkpoint file.")
            sys.exit(1)
        
        with open(scenes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        initial_data = PipelineResult.from_dict(data)
    
    # Run summary generation stage
    try:
        result = pipeline.run(initial_data)
        print(f"Summary generation complete.")
        
        if result.summary_file:
            print(f"Markdown summary saved to: {result.summary_file}")
        
    except Exception as e:
        print(f"Summary generation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Ensure video file exists
    if not os.path.exists(args.video_path):
        print(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Run the appropriate pipeline based on the run mode
    if args.run_mode == "full":
        run_full_pipeline(args)
    elif args.run_mode == "detect":
        run_scene_detection(args)
    elif args.run_mode == "process":
        run_scene_processing(args)
    elif args.run_mode == "summarize":
        run_summary_generation(args)
    else:
        print(f"Unknown run mode: {args.run_mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
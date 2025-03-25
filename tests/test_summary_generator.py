#!/usr/bin/env python3
"""
Test script for the LLMSummaryGenerator stage of the video summarization pipeline.
This script tests the summary generation for a small set of processed scenes
using the multi-provider LLM generator.
"""

import os
import json
import argparse
from vid_summerizer import LLMSummaryGenerator, Scene


def test_summary_generator(test_output_dir, provider="anthropic"):
    """Test the LLMSummaryGenerator with sample processed scenes."""

    # Path for the test summary file
    test_summary_file = os.path.join(test_output_dir, f"test_summary_{provider}.md")

    # Load processed scenes from previous test or create sample
    processed_scenes_file = os.path.join(
        test_output_dir, "processed_scenes.json")

    if os.path.exists(processed_scenes_file):
        print(f"Loading processed scenes from {processed_scenes_file}")
        with open(processed_scenes_file, 'r') as f:
            scene_dicts = json.load(f)

        # Convert dict to Scene objects
        scenes = [Scene(
            scene_id=s["scene_id"],
            start=s["start"],
            end=s["end"],
            screenshot=s["screenshot"],
            transcript=s["transcript"]
        ) for s in scene_dicts]
    else:
        # Create a basic sample with mock data
        print("Sample processed scenes file not found, using mock data")
        mock_screenshot_dir = os.path.join(
            test_output_dir, "processor_output/screenshots")
        os.makedirs(mock_screenshot_dir, exist_ok=True)

        mock_screenshot_path = os.path.join(mock_screenshot_dir, "scene_1.jpg")

        # Create a blank image if the screenshot doesn't exist
        if not os.path.exists(mock_screenshot_path):
            try:
                import cv2
                import numpy as np
                cv2.imwrite(
                    mock_screenshot_path, np.ones(
                        (480, 640, 3), dtype=np.uint8) * 255)
                print(f"Created dummy screenshot at {mock_screenshot_path}")
            except ImportError:
                print("OpenCV not available, please ensure the screenshot file exists")

        scenes = [
            Scene(
                scene_id=1,
                start=10.0,
                end=20.0,
                screenshot=mock_screenshot_path,
                transcript="This is a sample transcript for testing the summary generator with different LLM providers."
            )
        ]

    # Process only the first scene to save on API costs during testing
    scenes_to_process = scenes[:1]
    print(f"Processing {len(scenes_to_process)} scenes for summary generation using {provider.upper()}")

    # Create and run LLMSummaryGenerator with specified provider
    try:
        # Set appropriate model for each provider
        if provider == "anthropic":
            model = "claude-3-7-sonnet-20250219"
        elif provider == "openai":
            model = "gpt-4o-2024-11-20"
        elif provider == "gemini":
            model = "gemini-2.0-flash"
        else:
            model = "claude-3-7-sonnet-20250219"  # Default
            
        generator = LLMSummaryGenerator(
            model=model,
            max_tokens=500,
            output_dir=test_output_dir,
            preferred_provider=provider
        )
        
        print(f"Active provider: {generator.active_provider}")
        print(f"Available providers: {', '.join(generator.available_generators)}")
        
        summary = generator.run(scenes_to_process)

        # Save the summary to a file
        with open(test_summary_file, 'w') as f:
            f.write(summary)

        print(f"Summary generated and saved to {test_summary_file}")
        print("\nSummary preview:")
        print("-----------------------------------")
        lines = summary.split('\n')
        preview_lines = 10  # Show first 10 lines of the summary
        for line in lines[:preview_lines]:
            print(line)
        if len(lines) > preview_lines:
            print("...")
        print("-----------------------------------")

        return summary

    except Exception as e:
        print(f"Error testing summary generator: {str(e)}")
        return None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Get the project root directory
    root_dir = Path(__file__).parent.parent

    # Import from conftest.py
    sys.path.insert(0, str(root_dir))
    from tests.conftest import TEST_OUTPUT_DIR
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test the LLM summary generator")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini"], 
                        default="anthropic", help="LLM provider to test")
    args = parser.parse_args()

    # Run the test
    test_summary_generator(TEST_OUTPUT_DIR, args.provider)

#!/usr/bin/env python3
"""
Script to run a single test for the video summarization pipeline.
Usage: python run_single_test.py [scene_detector|scene_processor|summary_generator] [--force]
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test fixtures and functions
from tests.conftest import TEST_VIDEO_PATH, TEST_OUTPUT_DIR, get_test_stage_status
from tests.test_scene_detector import test_scene_detector
from tests.test_scene_processor import test_scene_processor
from tests.test_summary_generator import test_summary_generator


def run_single_test(test_name, force=False, provider="anthropic"):
    """Run a single test by name with dependency checking."""

    print("=" * 50)
    print(f"RUNNING TEST: {test_name.upper()}")
    if test_name == "summary_generator":
        print(f"LLM PROVIDER: {provider.upper()}")
    print("=" * 50)

    # Check dependencies
    if not force and test_name != "scene_detector":
        if not get_test_stage_status(test_name):
            print(f"ERROR: Dependencies for test '{test_name}' not satisfied.")
            print(
                "Run previous tests first or use --force to bypass dependency checking.")
            return 1

    try:
        if test_name == "scene_detector":
            test_scene_detector(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

        elif test_name == "scene_processor":
            test_scene_processor(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

        elif test_name == "summary_generator":
            test_summary_generator(TEST_OUTPUT_DIR, provider)

        else:
            print(f"ERROR: Unknown test name: {test_name}")
            print("Available tests: scene_detector, scene_processor, summary_generator")
            return 1

        print("\n" + "=" * 50)
        print(f"TEST {test_name.upper()} COMPLETED SUCCESSFULLY")
        print("=" * 50)

    except Exception as e:
        print(f"\nERROR: Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single test stage")
    parser.add_argument(
        "test_name",
        choices=[
            "scene_detector",
            "scene_processor",
            "summary_generator"],
        help="Name of the test stage to run")
    parser.add_argument(
        "--force", action="store_true",
        help="Force test execution even if dependencies aren't satisfied")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "gemini"],
        default="anthropic", help="LLM provider to use for summary generation")

    args = parser.parse_args()
    sys.exit(run_single_test(args.test_name, args.force, args.provider))

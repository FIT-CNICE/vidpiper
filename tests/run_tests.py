#!/usr/bin/env python3
"""
Run all the test scripts for the video summarization pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test fixtures
from tests.conftest import TEST_VIDEO_PATH, TEST_OUTPUT_DIR

# Import test functions
from tests.test_scene_detector import test_scene_detector
from tests.test_scene_processor import test_scene_processor
from tests.test_summary_generator import test_summary_generator


def run_all_tests(run_unit_tests=False, llm_provider="anthropic"):
    """Run all test scripts in sequence."""

    print("=" * 50)
    print("RUNNING VIDEO SUMMARIZATION PIPELINE TESTS")
    print("=" * 50)

    # Ensure output directory exists
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    try:
        if run_unit_tests:
            print("\n\n" + "=" * 50)
            print("RUNNING UNIT TESTS")
            print("=" * 50)
            import pytest

            # Run unit tests with pytest
            unit_test_files = [
                os.path.join(Path(__file__).parent, "test_llm_generators.py"),
                os.path.join(Path(__file__).parent, "test_pipeline.py"),
            ]

            for test_file in unit_test_files:
                print(f"\nRunning tests in {os.path.basename(test_file)}")
                pytest_result = pytest.main(["-xvs", test_file])
                if pytest_result != 0:
                    print(
                        f"Unit tests in {os.path.basename(test_file)} failed!"
                    )
                    return 1

        # Test 1: Scene Detection
        print("\n\n" + "=" * 50)
        print("TEST 1: SCENE DETECTION")
        print("=" * 50)
        test_scene_detector(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

        # Test 2: Scene Processing (Screenshots and Transcripts)
        print("\n\n" + "=" * 50)
        print("TEST 2: SCENE PROCESSING")
        print("=" * 50)
        test_scene_processor(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

        # Test 3: Summary Generation with selected LLM provider
        print("\n\n" + "=" * 50)
        print(
            f"TEST 3: SUMMARY GENERATION (Using {llm_provider.upper()} provider)"
        )
        print("=" * 50)
        test_summary_generator(TEST_OUTPUT_DIR, llm_provider)

        print("\n\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 50)

    except Exception as e:
        print(f"\n\nERROR: Test suite failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run tests for video summarization pipeline"
    )
    parser.add_argument(
        "--unit-tests",
        action="store_true",
        help="Run unit tests for LLM generators and Pipeline architecture",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini"],
        default="anthropic",
        help="LLM provider to test",
    )

    args = parser.parse_args()

    sys.exit(run_all_tests(args.unit_tests, args.provider))

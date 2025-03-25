"""
Configuration file for pytest and standalone test scripts.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Define constants for testing
# Use absolute paths for better reliability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_VIDEO_PATH = os.path.join(
    BASE_DIR,
    # "vid/an-ai-team-lead-guide.mp4")
    # "vid/distributed-light-baking-system-powered-by-optix-7.mp4"
    "vid/step-6.mp4")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "tests/output")

# Create output directories
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(TEST_OUTPUT_DIR, "processor_output"), exist_ok=True)
os.makedirs(
    os.path.join(
        TEST_OUTPUT_DIR,
        "processor_output/screenshots"),
    exist_ok=True)


@pytest.fixture
def test_video_path():
    """Return the path to the test video."""
    return TEST_VIDEO_PATH


@pytest.fixture
def test_output_dir():
    """Return the path to the test output directory."""
    return TEST_OUTPUT_DIR


# Function to get status of previous test stages (for dependency handling)
def get_test_stage_status(stage_name):
    """Check if a previous test stage has completed successfully."""
    if stage_name == "scene_detector":
        # No dependencies
        return True

    elif stage_name == "scene_processor":
        # Check if scene detection output exists
        return os.path.exists(os.path.join(
            TEST_OUTPUT_DIR, "scenes_test.json"))

    elif stage_name == "summary_generator":
        # Check if processed scenes exist
        return os.path.exists(os.path.join(
            TEST_OUTPUT_DIR, "processed_scenes.json"))

    return False
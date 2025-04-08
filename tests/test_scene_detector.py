#!/usr/bin/env python3
"""
Test script for the SceneDetector stage of the video summarization pipeline.
This script tests scene detection on a sample video from the vid/ folder.
"""

import os
import json
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_summarizer.core import PipelineResult
from video_summarizer.stages import create_scene_detector


def test_scene_detector(test_video_path, test_output_dir):
    """Test the SceneDetector stage with different thresholds."""

    # Verify the test video exists
    if not os.path.exists(test_video_path):
        print(f"ERROR: Test video not found at: {test_video_path}")
        return
    else:
        print(f"Found test video: {test_video_path}")

    # Using higher thresholds to detect fewer scenes and speed up testing
    thresholds = [45.0]

    for threshold in thresholds:
        print(f"\nTesting SceneDetector with threshold={threshold}")

        # Create and run SceneDetector with higher downscale factor for faster processing
        detector = create_scene_detector(
            threshold=threshold,
            # Much higher downscale factor for testing - uses less memory and runs faster
            downscale_factor=128,
            skip_start=0.0,  # Video is short, don't skip anything
            skip_end=0.0,  # Video is short, don't skip anything
        )

        # Create initial pipeline data
        initial_data = PipelineResult(
            video_path=os.path.abspath(test_video_path)
        )

        # Run the detector
        result = detector.run(initial_data)
        scenes = result.scenes

        # Print results
        print(f"Detected {len(scenes)} scenes:")
        for i, scene in enumerate(scenes[:5]):  # Print first 5 scenes
            print(
                f"  Scene {scene.scene_id}: {scene.start:.2f}s - {scene.end:.2f}s"
            )

        if len(scenes) > 5:
            print(f"  ... and {len(scenes) - 5} more scenes")

        # Convert scenes to dicts for JSON serialization (deprecated approach, now using PipelineResult.to_dict())
        scene_dicts = []
        for scene in scenes:
            scene_dicts.append(
                {
                    "scene_id": scene.scene_id,
                    "start": scene.start,
                    "end": scene.end,
                }
            )

        # Save results to a JSON file
        result_file = os.path.join(test_output_dir, "scenes_test.json")
        with open(result_file, "w") as f:
            json.dump(scene_dicts, f, indent=2)

        print(f"Results saved to {result_file}")

        # Also save using the new PipelineResult serialization
        new_result_file = os.path.join(
            test_output_dir, "scenes_test_pipeline_result.json"
        )
        with open(new_result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Full pipeline result saved to {new_result_file}")


if __name__ == "__main__":
    # Import from conftest.py
    from tests.conftest import TEST_VIDEO_PATH, TEST_OUTPUT_DIR

    # Create output dir
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Run the test
    test_scene_detector(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

#!/usr/bin/env python3
"""
Test script for the SceneProcessor stage of the video summarization pipeline.
This script tests screenshot extraction and transcript generation on a sample video.
"""

import os
import json
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_summarizer.core import Scene, PipelineResult
from video_summarizer.stages import create_scene_processor


def test_scene_processor(test_video_path, test_output_dir):
    """Test the SceneProcessor stage with sample scenes."""

    # Create processor output directory
    processor_output_dir = os.path.join(test_output_dir, "processor_output")
    os.makedirs(processor_output_dir, exist_ok=True)

    # Load or create sample scenes
    sample_scenes_file = os.path.join(test_output_dir, "scenes_test.json")
    pipeline_result_file = os.path.join(
        test_output_dir, "scenes_test_pipeline_result.json"
    )

    if os.path.exists(pipeline_result_file):
        print(f"Loading PipelineResult from {pipeline_result_file}")
        with open(pipeline_result_file, "r") as f:
            data = json.load(f)
        # Convert to PipelineResult
        result = PipelineResult.from_dict(data)
        scenes = result.scenes

    elif os.path.exists(sample_scenes_file):
        print(f"Loading sample scenes from {sample_scenes_file}")
        with open(sample_scenes_file, "r") as f:
            scene_dicts = json.load(f)
        # Convert dict to Scene objects
        scenes = [
            Scene(scene_id=s["scene_id"], start=s["start"], end=s["end"])
            for s in scene_dicts
        ]

        # Create a PipelineResult with the scenes
        result = PipelineResult(video_path=test_video_path, scenes=scenes)
    else:
        # Create some sample scenes manually
        print("Sample scenes file not found, creating sample scenes")
        scenes = [
            Scene(scene_id=1, start=10.0, end=20.0),
            Scene(scene_id=2, start=30.0, end=40.0),
            Scene(scene_id=3, start=50.0, end=60.0),
        ]

        # Create a PipelineResult with the scenes
        result = PipelineResult(video_path=test_video_path, scenes=scenes)

    # Process only the first few scenes to save time
    if len(scenes) > 30:
        scenes_to_process = scenes[:30]
        result.scenes = scenes_to_process
        print(
            f"Processing {len(scenes_to_process)} out of {len(scenes)} scenes"
        )
    else:
        print(f"Processing all {len(scenes)} scenes")

    # Create and run SceneProcessor
    processor = create_scene_processor(
        output_dir=processor_output_dir, use_whisper=True
    )

    # Process the scenes
    processed_result = processor.run(result)
    processed_scenes = processed_result.scenes

    # Print results
    print("\nProcessed scenes:")
    for scene in processed_scenes:
        print(f"Scene {scene.scene_id}:")
        print(f"  Screenshot: {scene.screenshot}")
        print(
            f"  Transcript: {scene.transcript[:50]}..."
            if len(scene.transcript or "") > 50
            else f"  Transcript: {scene.transcript}"
        )
        print()

    # Save processed scenes to a JSON file
    result_file = os.path.join(test_output_dir, "processed_scenes.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(processed_result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Results saved to {result_file}")
    print(f"Screenshots saved to {processor_output_dir}/screenshots/")


if __name__ == "__main__":
    # Import from conftest.py
    from tests.conftest import TEST_VIDEO_PATH, TEST_OUTPUT_DIR

    # Run the test
    test_scene_processor(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

#!/usr/bin/env python3
"""
Test script for the SceneProcessor stage of the video summarization pipeline.
This script tests screenshot extraction and transcript generation on a sample video.
"""

import os
import json
from vid_summerizer import SceneProcessor, Scene


def test_scene_processor(test_video_path, test_output_dir):
    """Test the SceneProcessor stage with sample scenes."""

    # Create processor output directory
    processor_output_dir = os.path.join(test_output_dir, "processor_output")
    os.makedirs(processor_output_dir, exist_ok=True)

    # Load or create sample scenes
    sample_scenes_file = os.path.join(
        test_output_dir, "scenes_test.json")

    if os.path.exists(sample_scenes_file):
        print(f"Loading sample scenes from {sample_scenes_file}")
        with open(sample_scenes_file, 'r') as f:
            scene_dicts = json.load(f)
        # Convert dict to Scene objects
        scenes = [Scene(
            scene_id=s["scene_id"],
            start=s["start"],
            end=s["end"]
        ) for s in scene_dicts]
    else:
        # Create some sample scenes manually
        print("Sample scenes file not found, creating sample scenes")
        scenes = [
            Scene(scene_id=1, start=10.0, end=20.0),
            Scene(scene_id=2, start=30.0, end=40.0),
            Scene(scene_id=3, start=50.0, end=60.0)
        ]

    # Process only the first 3 scenes to save time
    scenes_to_process = scenes[:30] if len(scenes) > 30 else scenes
    print(f"Processing {len(scenes_to_process)} scenes")

    # Create and run SceneProcessor
    processor = SceneProcessor(
        video_path=test_video_path,
        output_dir=processor_output_dir)
    processed_scenes = processor.run(scenes_to_process)

    # Print results
    print("\nProcessed scenes:")
    for scene in processed_scenes:
        print(f"Scene {scene.scene_id}:")
        print(f"  Screenshot: {scene.screenshot}")
        print(
            f"  Transcript: {scene.transcript[:50]}..."
            if len(scene.transcript or '') > 50 else
            f"  Transcript: {scene.transcript}")
        print()

    # Convert to dict for JSON serialization
    scene_dicts = []
    for scene in processed_scenes:
        scene_dicts.append({
            "scene_id": scene.scene_id,
            "start": scene.start,
            "end": scene.end,
            "screenshot": scene.screenshot,
            "transcript": scene.transcript
        })

    # Save processed scenes to a JSON file
    result_file = os.path.join(test_output_dir, "processed_scenes.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(scene_dicts, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {result_file}")
    print(f"Screenshots saved to {processor_output_dir}/screenshots/")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Get the project root directory
    root_dir = Path(__file__).parent.parent

    # Import from conftest.py
    sys.path.insert(0, str(root_dir))
    from tests.conftest import TEST_VIDEO_PATH, TEST_OUTPUT_DIR

    # Run the test
    test_scene_processor(TEST_VIDEO_PATH, TEST_OUTPUT_DIR)

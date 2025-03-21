#!/usr/bin/env python3
"""
This script creates mock data for testing without requiring the actual video or API calls.
Useful for testing the infrastructure or the final stage with synthetic data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test fixtures and utilities
from tests.conftest import TEST_OUTPUT_DIR
from tests.test_utils import create_mock_scenes, save_scenes_to_json

def create_mock_data(num_scenes=5, stage="all"):
    """Create mock data for testing."""
    
    print(f"Creating mock data for testing stage: {stage}")
    print(f"Output directory: {TEST_OUTPUT_DIR}")
    
    if stage == "scene_detector" or stage == "all":
        # Create mock scene detection output
        print("\nCreating mock scene detection data...")
        scenes = create_mock_scenes(num_scenes, TEST_OUTPUT_DIR)
        
        # Save basic scene data (just ID, start, end)
        scene_data = []
        for scene in scenes:
            scene_data.append({
                "scene_id": scene.scene_id,
                "start": scene.start,
                "end": scene.end
            })
        
        # Save scene data for each threshold
        for threshold in [20.0, 30.0, 40.0]:
            output_path = os.path.join(TEST_OUTPUT_DIR, f"scenes_threshold_{threshold}.json")
            with open(output_path, 'w') as f:
                import json
                json.dump(scene_data, f, indent=2)
            print(f"Scene data saved to {output_path}")
    
    if stage == "scene_processor" or stage == "all":
        # Create mock scene processor output
        print("\nCreating mock scene processor data...")
        scenes = create_mock_scenes(num_scenes, TEST_OUTPUT_DIR)
        
        # Save processed scenes
        output_path = os.path.join(TEST_OUTPUT_DIR, "processed_scenes.json")
        save_scenes_to_json(scenes, output_path)
        print(f"Processed scene data saved to {output_path}")
    
    if stage == "summary_generator" or stage == "all":
        # Create mock summary
        print("\nCreating mock summary data...")
        summary = "# Mock Video Summary\n\n"
        
        scenes = create_mock_scenes(num_scenes, TEST_OUTPUT_DIR)
        
        for scene in scenes:
            summary += f"## Scene {scene.scene_id} - [{int(scene.start // 60):02d}:{int(scene.start % 60):02d}]\n\n"
            summary += f"![Scene {scene.scene_id} Screenshot]({scene.screenshot})\n\n"
            summary += f"This is a mock summary for scene {scene.scene_id}. "
            summary += f"The scene shows a technical presentation about AI technologies.\n\n"
            summary += "Key points covered in this segment:\n"
            summary += "- Technical concept #1\n"
            summary += "- Important metric: 95% accuracy\n"
            summary += "- Main argument of the presenter\n\n"
            summary += "---\n\n"
        
        # Save summary
        output_path = os.path.join(TEST_OUTPUT_DIR, "test_summary.md")
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"Mock summary saved to {output_path}")
    
    print("\nMock data creation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mock data for testing")
    parser.add_argument("--num-scenes", type=int, default=5,
                        help="Number of mock scenes to create")
    parser.add_argument("--stage", choices=["all", "scene_detector", "scene_processor", "summary_generator"],
                        default="all", help="Test stage to create mock data for")
    
    args = parser.parse_args()
    create_mock_data(args.num_scenes, args.stage)
"""
Utility functions for testing the video summarization pipeline.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

from vid_summerizer import Scene

def create_mock_scene(scene_id, start_time, end_time, test_output_dir):
    """Create a mock scene with dummy screenshot and transcript."""
    
    # Create a directory for screenshots if it doesn't exist
    mock_screenshot_dir = os.path.join(test_output_dir, "processor_output/screenshots")
    os.makedirs(mock_screenshot_dir, exist_ok=True)
    
    # Create a mock screenshot file
    mock_screenshot_path = os.path.join(mock_screenshot_dir, f"scene_{scene_id}.jpg")
    
    # Create a dummy image with some text to make it more realistic
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some text
    cv2.putText(img, f"Test Scene {scene_id}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Time: {start_time:.2f}s - {end_time:.2f}s", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Write the image
    cv2.imwrite(mock_screenshot_path, img)
    
    # Create a mock transcript
    mock_transcript = f"This is a sample transcript for test scene {scene_id}, " \
                    f"which runs from {start_time:.2f} to {end_time:.2f} seconds."
    
    # Return a Scene object
    return Scene(
        scene_id=scene_id,
        start=start_time,
        end=end_time,
        screenshot=mock_screenshot_path,
        transcript=mock_transcript
    )

def create_mock_scenes(num_scenes, test_output_dir):
    """Create a set of mock scenes for testing."""
    
    scenes = []
    scene_duration = 10.0  # Duration of each scene in seconds
    
    for i in range(1, num_scenes + 1):
        start_time = (i - 1) * scene_duration
        end_time = i * scene_duration
        
        scenes.append(create_mock_scene(i, start_time, end_time, test_output_dir))
    
    return scenes

def save_scenes_to_json(scenes, output_path, as_dict=True):
    """Save scene data to a JSON file."""
    
    if as_dict:
        # Convert Scene objects to dictionaries
        scene_dicts = []
        for scene in scenes:
            scene_dicts.append({
                "scene_id": scene.scene_id,
                "start": scene.start,
                "end": scene.end,
                "screenshot": scene.screenshot,
                "transcript": scene.transcript
            })
        data = scene_dicts
    else:
        # Keep as Scene objects (not JSON serializable, but useful for debugging)
        data = scenes
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

def load_scenes_from_json(json_path):
    """Load scene data from a JSON file and convert to Scene objects."""
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        scene_dicts = json.load(f)
    
    # Convert dictionaries to Scene objects
    scenes = []
    for s in scene_dicts:
        scenes.append(Scene(
            scene_id=s["scene_id"],
            start=s["start"],
            end=s["end"],
            screenshot=s.get("screenshot"),
            transcript=s.get("transcript")
        ))
    
    return scenes
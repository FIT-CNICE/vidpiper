#!/usr/bin/env python3
"""
Test script for the faster-whisper transcription in SceneProcessor.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidpiper.stages.scene_processor import SceneProcessor


def test_placeholder_transcript():
    """Test the _placeholder_transcript function with a real video file."""

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = SceneProcessor(output_dir=temp_dir)

        # Try to import faster-whisper
        try:
            from faster_whisper import WhisperModel  # noqa: F401

            print("faster-whisper imported successfully")
        except ImportError:
            print("ERROR: faster-whisper not found")
            return

        # Test with a non-existent video path to check error handling
        test_error_handling = False

        if test_error_handling:
            video_path = "/path/to/nonexistent/video.mp4"
            print("Testing error handling with a non-existent video path")
        else:
            # Use the demo video file from cdeus_demo
            video_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "cdeus_demo",
                "dgx.mp4",
            )

        # For the error handling test, we'll still run the function
        # even if the file doesn't exist
        if not os.path.exists(video_path) and not test_error_handling:
            print(f"ERROR: Video file not found at {video_path}")
            return

        print(f"Testing with video file: {video_path}")

        # Call the _placeholder_transcript function directly
        # Test with a longer segment (10-20 seconds) of the video
        start_time = 10.0
        end_time = 20.0

        print(f"Transcribing segment from {start_time}s to {end_time}s...")
        transcript = processor._placeholder_transcript(
            video_path, start_time, end_time
        )

        print(f"\nTranscription result:\n{transcript}\n")

        # Cleanup
        processor.cleanup()


if __name__ == "__main__":
    print("Testing the faster-whisper CPU transcription...")
    test_placeholder_transcript()

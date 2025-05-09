#!/usr/bin/env python3
"""
Test script for the SummaryUpdater stage.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidpiper.stages import create_summary_updater


def test_summary_updater():
    """Test the SummaryUpdater with a sample summary and feedback."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample summary file
        sample_summary = """# Video Summary

## Scene 1 - [00:05]

![Scene 1 Screenshot](./screenshots/scene_1.jpg)

In this scene, the presenter introduces the new AI hardware accelerator. The key features mentioned include:
- 30% faster inference compared to previous generation
- Support for quantized models (INT8, INT4)
- Lower power consumption (under 15W TDP)

## Scene 2 - [01:20]

![Scene 2 Screenshot](./screenshots/scene_2.jpg)

The presenter demonstrates a real-time object detection application running on the new hardware. The demo shows the system processing a 4K video stream at 60 frames per second while identifying and tracking multiple objects.
"""

        # Create sample feedback
        sample_feedback = """
Please make the following improvements to the summary:
1. In Scene 1, add that the hardware accelerator supports both transformer and CNN architectures
2. For Scene 2, mention that the system is using less than 10W of power during the demo
3. Add more technical precision about the object detection - it's using YOLOv8 with 99.2% accuracy
"""

        # Write the sample summary to a file
        summary_path = os.path.join(temp_dir, "test_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(sample_summary)

        print(f"Created test summary at: {summary_path}")

        # Create a summary updater
        updater = create_summary_updater(output_dir=temp_dir)

        # Update the summary
        try:
            updated_path = updater.update_summary_file(
                summary_file=summary_path, feedback=sample_feedback
            )

            # Read the updated summary
            with open(updated_path, "r", encoding="utf-8") as f:
                updated_summary = f.read()

            # Check if the feedback was incorporated (case-insensitive partial matches)
            success = all(
                [
                    "transformer" in updated_summary.lower()
                    and "cnn" in updated_summary.lower(),
                    "10w" in updated_summary.lower()
                    and "power" in updated_summary.lower(),
                    "yolov8" in updated_summary.lower(),
                    "99.2" in updated_summary
                    and "accuracy" in updated_summary.lower(),
                ]
            )

            if success:
                print(
                    "✅ Test passed! All feedback was incorporated into the updated summary."
                )
                print("\nUpdated summary content:")
                print("-" * 50)
                print(updated_summary)
                print("-" * 50)
            else:
                print("❌ Test failed! Not all feedback was incorporated.")
                print("\nUpdated summary content:")
                print("-" * 50)
                print(updated_summary)
                print("-" * 50)

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
        finally:
            updater.cleanup()


if __name__ == "__main__":
    print("Testing SummaryUpdater...")
    test_summary_updater()

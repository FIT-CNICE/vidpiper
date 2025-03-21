#!/usr/bin/env python3
"""
Run all the test scripts for the video summarization pipeline.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test fixtures
from tests.conftest import TEST_VIDEO_PATH, TEST_OUTPUT_DIR

# Import test functions
from tests.test_scene_detector import test_scene_detector
from tests.test_scene_processor import test_scene_processor
from tests.test_summary_generator import test_summary_generator

def run_all_tests():
    """Run all test scripts in sequence."""
    
    print("=" * 50)
    print("RUNNING VIDEO SUMMARIZATION PIPELINE TESTS")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    try:
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
        
        # Test 3: Summary Generation (Claude API)
        print("\n\n" + "=" * 50)
        print("TEST 3: SUMMARY GENERATION")
        print("=" * 50)
        test_summary_generator(TEST_OUTPUT_DIR)
        
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
    sys.exit(run_all_tests())
# Test just scene detection

./gtc/tests/run_single_test.py scene_detector

# Test just scene processing

./gtc/tests/run_single_test.py scene_processor

# Test just summary generation

./gtc/tests/run_single_test.py summary_generator

# Generate mock data to test without a real video

./gtc/tests/mock_test.py --stage scene_processor

The tests have dependency checks, so they will ensure that previous stages have
completed before running. You can bypass this with the --force flag.

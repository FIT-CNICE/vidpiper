# Video Summarizer Test Suite

This directory contains tests for the video summarization pipeline. The tests verify each stage of the pipeline: scene detection, scene processing, and summary generation.

## Running Tests

### Test Individual Stages

```bash
# Test just scene detection
./gtc/tests/run_single_test.py scene_detector

# Test just scene processing
./gtc/tests/run_single_test.py scene_processor

# Test just summary generation
./gtc/tests/run_single_test.py summary_generator

# Test summary generation with a specific LLM provider
./gtc/tests/run_single_test.py summary_generator --provider openai
```

### Run All Tests

```bash
# Run all tests with default settings (using Anthropic)
./gtc/tests/run_tests.py

# Run all tests including unit tests
./gtc/tests/run_tests.py --unit-tests

# Run all tests with a specific LLM provider
./gtc/tests/run_tests.py --provider openai
```

### Generate Mock Data

Generate mock data to test without a real video:

```bash
# Generate mock data for all stages
./gtc/tests/mock_test.py

# Generate mock data for a specific stage
./gtc/tests/mock_test.py --stage scene_processor

# Generate a specific number of mock scenes
./gtc/tests/mock_test.py --num-scenes 10
```

## Test Dependencies

The tests have dependency checks, so they will ensure that previous stages have completed before running. You can bypass this with the `--force` flag.

## Unit Tests

The `test_llm_generators.py` file contains unit tests for the LLM generator classes. Run these tests to verify:

- Each provider generator (Anthropic, OpenAI, Gemini) works correctly
- The LLMSummaryGenerator switches providers properly
- Error handling and fallback mechanisms work as expected

Run the unit tests with:

```bash
python -m pytest -xvs ./gtc/tests/test_llm_generators.py
```

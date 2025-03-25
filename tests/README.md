# Video Summarizer Test Suite

This directory contains tests for the video summarization pipeline. The tests verify each stage of the pipeline: scene detection, scene processing, and summary generation.

## Test Organization

- **test_pipeline.py**: Tests for the core Pipeline architecture
- **test_scene_detector.py**: Tests for the scene detection stage
- **test_scene_processor.py**: Tests for the screenshot extraction and transcript generation
- **test_summary_generator.py**: Tests for the summary generation with LLMs
- **test_llm_generators.py**: Unit tests for the LLM provider classes
- **run_tests.py**: Script to run all tests in sequence
- **run_single_test.py**: Script to run a specific test stage
- **conftest.py**: Pytest configuration and shared fixtures

## Running Tests

There are multiple ways to run the tests:

### Test Individual Stages

```bash
# Test just scene detection
./tests/run_single_test.py scene_detector

# Test just scene processing
./tests/run_single_test.py scene_processor

# Test just summary generation
./tests/run_single_test.py summary_generator

# Test summary generation with a specific LLM provider
./tests/run_single_test.py summary_generator --provider openai
```

### Run All Tests

```bash
# Run all tests with default settings (using Anthropic)
./tests/run_tests.py

# Run all tests including unit tests
./tests/run_tests.py --unit-tests

# Run all tests with a specific LLM provider
./tests/run_tests.py --provider openai
```

### Run Individual Test Modules

```bash
# Run a specific test module
python tests/test_scene_detector.py
python tests/test_scene_processor.py
python tests/test_summary_generator.py --provider anthropic
```

### Run Unit Tests with Pytest

```bash
# Run all unit tests
pytest tests/test_pipeline.py tests/test_llm_generators.py

# Run specific test file
pytest tests/test_pipeline.py

# Run specific test class
pytest tests/test_pipeline.py::TestPipeline

# Run specific test method
pytest tests/test_pipeline.py::TestPipeline::test_pipeline_execution
```

## Test Data

Tests use a sample video from the `vid/` directory and store output files in the `tests/output/` directory.

## Test Dependencies

The tests have dependency checks, so they will ensure that previous stages have completed before running. You can bypass this with the `--force` flag.

## Unit Tests

The `test_llm_generators.py` file contains unit tests for the LLM generator classes and `test_pipeline.py` contains tests for the pipeline architecture. Run these tests to verify:

- Each provider generator (Anthropic, OpenAI, Gemini) works correctly
- The LLMSummaryGenerator switches providers properly
- Error handling and fallback mechanisms work as expected
- The pipeline architecture works correctly with different stages
- Stage manipulation (add, insert, replace, remove) works properly

## Adding New Tests

When adding new components to the video summarizer pipeline:

1. Create a new test file following the naming convention `test_component_name.py`
2. Add unit tests for individual functions and classes
3. Add integration tests that verify interaction with other components
4. Update `run_tests.py` to include your new test file if needed
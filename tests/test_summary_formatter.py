"""Tests for the SummaryFormatter stage."""
import os
import pytest
import tempfile
import shutil
from pathlib import Path

from video_summarizer.core.data_classes import PipelineResult
from video_summarizer.stages import SummaryFormatter, create_summary_formatter


@pytest.fixture
def sample_summary():
    """Generate a sample summary file for testing."""
    content = """# Video Summary

## Scene 1 - [00:10]

![Scene 1 Screenshot](./screenshots/scene1.jpg)

The presenter introduces the key features of the new product.

## Scene 2 - [01:20]

![Scene 2 Screenshot](./screenshots/scene2.jpg)

A live demonstration of the product's capabilities.
"""
    return content


@pytest.fixture
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    # Create a screenshots directory
    os.makedirs(os.path.join(temp_dir, "screenshots"), exist_ok=True)
    
    # Yield the temporary directory
    yield temp_dir
    
    # Clean up after the test
    shutil.rmtree(temp_dir)


def test_summary_formatter_single_file(test_dir, sample_summary, monkeypatch):
    """Test that the SummaryFormatter can format a single summary file."""
    # Mock the LLM generation to avoid API calls
    def mock_generate_content(self, prompt, image_data=None):
        return """---BEGIN MARP DECK---
---
marp: true
title: Technical Presentation
subtitle: Product Introduction and Demo
author: AI Formatter
taxonomy: Technology > Software > Product Demo
---

# Product Introduction
![bg right:40%](./screenshots/scene1.jpg)

- Key features of the new product
- Introduced at timestamp [00:10]

---

# Live Demonstration
![bg right:40%](./screenshots/scene2.jpg)

- Showcasing product capabilities
- Demonstrated at timestamp [01:20]
---END MARP DECK---"""
    
    # Apply the mock
    monkeypatch.setattr(
        "video_summarizer.llm_providers.GeminiGenerator.generate_content", 
        mock_generate_content
    )
    monkeypatch.setattr(
        "video_summarizer.llm_providers.get_available_llm_providers", 
        lambda: {"gemini": True, "anthropic": False, "openai": False}
    )
    
    # Create a summary file
    summary_path = os.path.join(test_dir, "test_sum.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(sample_summary)
    
    # Create a dummy image file
    screenshot_path = os.path.join(test_dir, "screenshots", "scene1.jpg")
    with open(screenshot_path, "w", encoding="utf-8") as f:
        f.write("dummy image data")
    
    screenshot_path = os.path.join(test_dir, "screenshots", "scene2.jpg")
    with open(screenshot_path, "w", encoding="utf-8") as f:
        f.write("dummy image data")
    
    # Create the formatter
    formatter = create_summary_formatter(output_dir=test_dir)
    
    # Set up pipeline data
    data = PipelineResult(
        video_path="/dummy/path/video.mp4",
        summary_file=summary_path,
        output_dir=test_dir
    )
    
    # Run the formatter
    result = formatter.run(data)
    
    # Check that the formatted file was created
    assert result.formatted_file is not None
    assert os.path.exists(result.formatted_file)
    
    # Check that the formatted file has the expected name
    expected_name = "test_fmt.md"
    assert os.path.basename(result.formatted_file) == expected_name
    
    # Check the formatted file content
    with open(result.formatted_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "marp: true" in content
        assert "title: Technical Presentation" in content
        assert "taxonomy: Technology > Software > Product Demo" in content


def test_summary_formatter_directory(test_dir, sample_summary, monkeypatch):
    """Test that the SummaryFormatter can process a directory of summary files."""
    # Create subdirectories
    subdir1 = os.path.join(test_dir, "subdir1")
    subdir2 = os.path.join(test_dir, "subdir2")
    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)
    os.makedirs(os.path.join(subdir1, "screenshots"), exist_ok=True)
    os.makedirs(os.path.join(subdir2, "screenshots"), exist_ok=True)
    
    # Create summary files
    summary_path1 = os.path.join(subdir1, "test1_sum.md")
    summary_path2 = os.path.join(subdir2, "test2_sum.md")
    
    with open(summary_path1, "w", encoding="utf-8") as f:
        f.write(sample_summary)
    with open(summary_path2, "w", encoding="utf-8") as f:
        f.write(sample_summary)
    
    # Mock the LLM generation to avoid API calls
    def mock_generate_content(self, prompt, image_data=None):
        return """---BEGIN MARP DECK---
---
marp: true
title: Formatted Presentation
subtitle: Test
author: AI Formatter
taxonomy: Test > Example > Demo
---

# Test Slide
---END MARP DECK---"""
    
    # Apply the mock
    monkeypatch.setattr(
        "video_summarizer.llm_providers.GeminiGenerator.generate_content", 
        mock_generate_content
    )
    monkeypatch.setattr(
        "video_summarizer.llm_providers.get_available_llm_providers", 
        lambda: {"gemini": True, "anthropic": False, "openai": False}
    )
    
    # Create the formatter
    formatter = create_summary_formatter(output_dir=test_dir)
    
    # Process the directory directly using the standalone method
    formatted_files = formatter.format_directory(test_dir)
    
    # Check that both files were formatted
    assert len(formatted_files) == 2
    
    # Check that both formatted files exist
    for formatted_file in formatted_files:
        assert os.path.exists(formatted_file)
        with open(formatted_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "marp: true" in content


def test_link_expansion(test_dir, monkeypatch):
    """Test that relative links are expanded to absolute paths."""
    # Create a summary with relative links
    summary_with_links = """# Summary with Links

![Relative Image](./screenshots/image.jpg)

[Link to another file](./another_file.md)
"""
    
    summary_path = os.path.join(test_dir, "links_sum.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_with_links)
    
    # Create the referenced files
    os.makedirs(os.path.join(test_dir, "screenshots"), exist_ok=True)
    with open(os.path.join(test_dir, "screenshots", "image.jpg"), "w") as f:
        f.write("dummy image")
    
    with open(os.path.join(test_dir, "another_file.md"), "w") as f:
        f.write("another file")
    
    # Mock the LLM generation to return the expanded links
    def mock_expand_links(self, content, base_dir):
        # Just making sure the method was called - return a marker
        return content + "\n<!-- Links expanded -->"
    
    # Apply the mock
    monkeypatch.setattr(
        "video_summarizer.stages.summary_formatter.SummaryFormatter._expand_links", 
        mock_expand_links
    )
    
    # Also mock the LLM generation
    def mock_generate_content(self, prompt, image_data=None):
        if "<!-- Links expanded -->" in prompt:
            return "---BEGIN MARP DECK---\nLinks were expanded\n---END MARP DECK---"
        else:
            return "---BEGIN MARP DECK---\nLinks were NOT expanded\n---END MARP DECK---"
    
    monkeypatch.setattr(
        "video_summarizer.llm_providers.GeminiGenerator.generate_content", 
        mock_generate_content
    )
    monkeypatch.setattr(
        "video_summarizer.llm_providers.get_available_llm_providers", 
        lambda: {"gemini": True, "anthropic": False, "openai": False}
    )
    
    # Create the formatter
    formatter = create_summary_formatter(output_dir=test_dir)
    
    # Process the file
    data = PipelineResult(
        video_path="/dummy/path/video.mp4",
        summary_file=summary_path,
        output_dir=test_dir
    )
    
    result = formatter.run(data)
    
    # Check that the formatted file was created
    assert os.path.exists(result.formatted_file)
    
    # Check the content
    with open(result.formatted_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Links were expanded" in content
#!/usr/bin/env python3
"""
Test script for the LLM Generator classes of the video summarization pipeline.
This script tests the different generators (Anthropic, OpenAI, Gemini) and the main LLMSummaryGenerator.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vid_summerizer import (
    Scene, 
    LLMGenerator,
    AnthropicGenerator, 
    OpenAIGenerator, 
    GeminiGenerator, 
    LLMSummaryGenerator
)


class TestAnthropicGenerator:
    """Tests for the AnthropicGenerator class."""
    
    def test_initialization(self):
        """Test that AnthropicGenerator initializes correctly."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                generator = AnthropicGenerator()
                
                # Check initialization
                assert generator.api_key == "test_key"
                assert generator.model == "claude-3-7-sonnet-20250219"
                assert generator.max_tokens == 2000
                assert generator.client is not None
                mock_anthropic.assert_called_once_with(api_key="test_key")
    
    def test_is_available(self):
        """Test the is_available method."""
        # Test when API key is available
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            with patch("anthropic.Anthropic"):
                generator = AnthropicGenerator()
                assert generator.is_available() is True
        
        # Test when API key is not available
        with patch.dict(os.environ, {}, clear=True):
            generator = AnthropicGenerator()
            assert generator.is_available() is False
    
    def test_generate_content_with_image(self):
        """Test generating content with an image."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            # Create a mock for the Anthropic client
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Generated text from Anthropic"
            mock_message.content = [mock_content]
            mock_client.messages.create.return_value = mock_message
            
            with patch("anthropic.Anthropic", return_value=mock_client):
                generator = AnthropicGenerator()
                
                # Test with image
                result = generator.generate_content("Test prompt", "test_image_data")
                
                # Verify result
                assert result == "Generated text from Anthropic"
                
                # Verify correct API call
                mock_client.messages.create.assert_called_once()
                call_args = mock_client.messages.create.call_args[1]
                assert call_args["model"] == "claude-3-7-sonnet-20250219"
                assert call_args["max_tokens"] == 2000
                assert len(call_args["messages"]) == 1
                
                # Verify the message has both image and text content
                message_content = call_args["messages"][0]["content"]
                assert len(message_content) == 2
                assert message_content[0]["type"] == "image"
                assert message_content[1]["type"] == "text"
                assert message_content[1]["text"] == "Test prompt"
    
    def test_generate_content_text_only(self):
        """Test generating content with text only."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            # Create a mock for the Anthropic client
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Generated text from Anthropic"
            mock_message.content = [mock_content]
            mock_client.messages.create.return_value = mock_message
            
            with patch("anthropic.Anthropic", return_value=mock_client):
                generator = AnthropicGenerator()
                
                # Test without image
                result = generator.generate_content("Test prompt", None)
                
                # Verify result
                assert result == "Generated text from Anthropic"
                
                # Verify correct API call
                call_args = mock_client.messages.create.call_args[1]
                assert call_args["model"] == "claude-3-7-sonnet-20250219"
                assert call_args["messages"][0]["content"] == "Test prompt"


class TestOpenAIGenerator:
    """Tests for the OpenAIGenerator class."""
    
    def test_initialization(self):
        """Test that OpenAIGenerator initializes correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch("openai.OpenAI") as mock_openai:
                generator = OpenAIGenerator()
                
                # Check initialization
                assert generator.api_key == "test_key"
                assert generator.model == "gpt-4-vision-preview"
                assert generator.max_tokens == 2000
                assert generator.client is not None
                mock_openai.assert_called_once_with(api_key="test_key")
    
    def test_is_available(self):
        """Test the is_available method."""
        # Test when API key is available
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch("openai.OpenAI"):
                generator = OpenAIGenerator()
                assert generator.is_available() is True
        
        # Test when API key is not available
        with patch.dict(os.environ, {}, clear=True):
            generator = OpenAIGenerator()
            assert generator.is_available() is False
    
    def test_generate_content_with_image(self):
        """Test generating content with an image."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            # Create a mock for the OpenAI client
            mock_client = MagicMock()
            mock_chat = MagicMock()
            mock_completions = MagicMock()
            mock_create = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Generated text from OpenAI"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response
            mock_completions.create = mock_create
            mock_chat.completions = mock_completions
            mock_client.chat = mock_chat
            
            with patch("openai.OpenAI", return_value=mock_client):
                generator = OpenAIGenerator()
                
                # Test with image
                result = generator.generate_content("Test prompt", "test_image_data")
                
                # Verify result
                assert result == "Generated text from OpenAI"
                
                # Verify correct API call
                mock_create.assert_called_once()
                call_args = mock_create.call_args[1]
                assert call_args["model"] == "gpt-4-vision-preview"
                assert call_args["max_tokens"] == 2000
                assert len(call_args["messages"]) == 2  # System message and user message
                
                # Verify the user message has both text and image
                user_message = call_args["messages"][1]
                assert user_message["role"] == "user"
                assert len(user_message["content"]) == 2
                assert user_message["content"][0]["type"] == "text"
                assert user_message["content"][0]["text"] == "Test prompt"
                assert user_message["content"][1]["type"] == "image_url"
                assert "data:image/jpeg;base64,test_image_data" in user_message["content"][1]["image_url"]["url"]
    
    def test_generate_content_text_only(self):
        """Test generating content with text only."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            # Create a mock for the OpenAI client
            mock_client = MagicMock()
            mock_chat = MagicMock()
            mock_completions = MagicMock()
            mock_create = MagicMock()
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            
            mock_message.content = "Generated text from OpenAI"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response
            mock_completions.create = mock_create
            mock_chat.completions = mock_completions
            mock_client.chat = mock_chat
            
            with patch("openai.OpenAI", return_value=mock_client):
                generator = OpenAIGenerator()
                
                # Test without image
                result = generator.generate_content("Test prompt", None)
                
                # Verify result
                assert result == "Generated text from OpenAI"
                
                # Verify correct API call
                call_args = mock_create.call_args[1]
                assert call_args["model"] == "gpt-4-vision-preview"
                assert len(call_args["messages"]) == 2
                assert call_args["messages"][1]["content"] == "Test prompt"


class TestGeminiGenerator:
    """Tests for the GeminiGenerator class."""
    
    def test_initialization(self):
        """Test that GeminiGenerator initializes correctly."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch("google.generativeai") as mock_genai:
                generator = GeminiGenerator()
                
                # Check initialization
                assert generator.api_key == "test_key"
                assert generator.model == "gemini-pro-vision"
                assert generator.max_tokens == 2000
                assert generator.genai is not None
                mock_genai.configure.assert_called_once_with(api_key="test_key")
    
    def test_is_available(self):
        """Test the is_available method."""
        # Test when API key is available
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch("google.generativeai"):
                generator = GeminiGenerator()
                assert generator.is_available() is True
        
        # Test when API key is not available
        with patch.dict(os.environ, {}, clear=True):
            generator = GeminiGenerator()
            assert generator.is_available() is False
    
    def test_generate_content_with_image(self):
        """Test generating content with an image."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Create mocks for Gemini
            mock_genai = MagicMock()
            mock_model = MagicMock()
            mock_response = MagicMock()
            
            mock_response.text = "Generated text from Gemini"
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            with patch("google.generativeai", mock_genai):
                with patch("google.generativeai.configure"):
                    with patch("base64.b64decode"):
                        with patch("PIL.Image.open"):
                            generator = GeminiGenerator()
                            generator.genai = mock_genai
                            
                            # Test with image
                            result = generator.generate_content("Test prompt", "test_image_data")
                            
                            # Verify result
                            assert result == "Generated text from Gemini"
                            
                            # Verify correct API call
                            mock_genai.GenerativeModel.assert_called_once_with("gemini-pro-vision")
                            mock_model.generate_content.assert_called_once()
                            # The generate_content is called with system prompt, user prompt, and image
                            assert len(mock_model.generate_content.call_args[0][0]) == 3
    
    def test_generate_content_text_only(self):
        """Test generating content with text only."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Create mocks for Gemini
            mock_genai = MagicMock()
            mock_model = MagicMock()
            mock_response = MagicMock()
            
            mock_response.text = "Generated text from Gemini"
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            with patch("google.generativeai", mock_genai):
                with patch("google.generativeai.configure"):
                    generator = GeminiGenerator()
                    generator.genai = mock_genai
                    
                    # Test without image
                    result = generator.generate_content("Test prompt", None)
                    
                    # Verify result
                    assert result == "Generated text from Gemini"
                    
                    # Verify correct API call - should use gemini-pro for text-only
                    mock_genai.GenerativeModel.assert_called_once_with("gemini-pro")
                    mock_model.generate_content.assert_called_once()
                    # The generate_content is called with system prompt and user prompt
                    assert len(mock_model.generate_content.call_args[0][0]) == 2


class TestLLMSummaryGenerator:
    """Tests for the LLMSummaryGenerator class."""
    
    def test_initialization_with_preferences(self):
        """Test initialization with different provider preferences."""
        # All providers available
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test_key", 
            "OPENAI_API_KEY": "test_key",
            "GEMINI_API_KEY": "test_key"
        }):
            # Mock all provider initializations
            with patch("vid_summerizer.AnthropicGenerator.is_available", return_value=True):
                with patch("vid_summerizer.OpenAIGenerator.is_available", return_value=True):
                    with patch("vid_summerizer.GeminiGenerator.is_available", return_value=True):
                        # Test anthropic preference
                        generator = LLMSummaryGenerator(preferred_provider="anthropic")
                        assert generator.active_provider == "anthropic"
                        assert set(generator.available_generators) == {"anthropic", "openai", "gemini"}
                        
                        # Test openai preference
                        generator = LLMSummaryGenerator(preferred_provider="openai")
                        assert generator.active_provider == "openai"
                        
                        # Test gemini preference
                        generator = LLMSummaryGenerator(preferred_provider="gemini")
                        assert generator.active_provider == "gemini"
    
    def test_initialization_with_limited_availability(self):
        """Test initialization when preferred provider is not available."""
        # Only OpenAI available
        with patch("vid_summerizer.AnthropicGenerator.is_available", return_value=False):
            with patch("vid_summerizer.OpenAIGenerator.is_available", return_value=True):
                with patch("vid_summerizer.GeminiGenerator.is_available", return_value=False):
                    # Prefer Anthropic, should fall back to OpenAI
                    generator = LLMSummaryGenerator(preferred_provider="anthropic")
                    assert generator.active_provider == "openai"
                    assert generator.available_generators == ["openai"]
    
    def test_initialization_no_providers(self):
        """Test initialization when no providers are available."""
        with patch("vid_summerizer.AnthropicGenerator.is_available", return_value=False):
            with patch("vid_summerizer.OpenAIGenerator.is_available", return_value=False):
                with patch("vid_summerizer.GeminiGenerator.is_available", return_value=False):
                    # Should raise ValueError
                    with pytest.raises(ValueError) as excinfo:
                        LLMSummaryGenerator()
                    assert "No LLM API keys found" in str(excinfo.value)
    
    def test_generate_scene_summary(self):
        """Test the _generate_scene_summary method."""
        # Mock available generators
        with patch("vid_summerizer.AnthropicGenerator.is_available", return_value=True):
            with patch("vid_summerizer.OpenAIGenerator.is_available", return_value=True):
                with patch("vid_summerizer.GeminiGenerator.is_available", return_value=False):
                    # Create mocks for content generation
                    mock_anthropic_generator = MagicMock()
                    mock_anthropic_generator.generate_content.return_value = "Summary from Anthropic"
                    mock_anthropic_generator.is_available.return_value = True
                    
                    mock_openai_generator = MagicMock()
                    mock_openai_generator.generate_content.return_value = "Summary from OpenAI"
                    mock_openai_generator.is_available.return_value = True
                    
                    # Create the generator with mocked providers
                    generator = LLMSummaryGenerator(preferred_provider="anthropic")
                    generator.generators = {
                        "anthropic": mock_anthropic_generator,
                        "openai": mock_openai_generator
                    }
                    generator.available_generators = ["anthropic", "openai"]
                    generator.active_provider = "anthropic"
                    
                    # Mock the _encode_image method
                    with patch.object(generator, "_encode_image", return_value="encoded_image"):
                        # Test successful generation with primary provider
                        result = generator._generate_scene_summary(
                            scene_id=1,
                            screenshot_path="test.jpg",
                            transcript="Test transcript",
                            start_time=0.0,
                            end_time=10.0,
                            scene_index=0,
                            total_scenes=3
                        )
                        
                        # Verify the result
                        assert result == "Summary from Anthropic"
                        mock_anthropic_generator.generate_content.assert_called_once()
    
    def test_generate_scene_summary_fallback(self):
        """Test fallback to another provider when the primary provider fails."""
        # Mock available generators
        with patch("vid_summerizer.AnthropicGenerator.is_available", return_value=True):
            with patch("vid_summerizer.OpenAIGenerator.is_available", return_value=True):
                with patch("vid_summerizer.GeminiGenerator.is_available", return_value=False):
                    # Create mocks for content generation
                    mock_anthropic_generator = MagicMock()
                    mock_anthropic_generator.generate_content.side_effect = Exception("Anthropic error")
                    mock_anthropic_generator.is_available.return_value = True
                    
                    mock_openai_generator = MagicMock()
                    mock_openai_generator.generate_content.return_value = "Summary from OpenAI"
                    mock_openai_generator.is_available.return_value = True
                    
                    # Create the generator with mocked providers
                    generator = LLMSummaryGenerator(preferred_provider="anthropic")
                    generator.generators = {
                        "anthropic": mock_anthropic_generator,
                        "openai": mock_openai_generator
                    }
                    generator.available_generators = ["anthropic", "openai"]
                    generator.active_provider = "anthropic"
                    
                    # Mock the _encode_image method and sleep to avoid actual waiting
                    with patch.object(generator, "_encode_image", return_value="encoded_image"):
                        with patch("time.sleep"):
                            # Test fallback when primary provider fails
                            result = generator._generate_scene_summary(
                                scene_id=1,
                                screenshot_path="test.jpg",
                                transcript="Test transcript",
                                start_time=0.0,
                                end_time=10.0,
                                scene_index=0,
                                total_scenes=3
                            )
                            
                            # Verify the result
                            assert result == "Summary from OpenAI"
                            assert mock_anthropic_generator.generate_content.call_count == 3  # Should retry 3 times
                            mock_openai_generator.generate_content.assert_called_once()
                            
                            # Check that active provider switched
                            assert generator.active_provider == "openai"
    
    def test_run_method(self):
        """Test the run method with a simple scene."""
        # Create a mock scene
        scene = Scene(
            scene_id=1,
            start=0.0,
            end=10.0,
            screenshot="/path/to/screenshot.jpg",
            transcript="Test transcript"
        )
        
        # Mock the generator initialization
        with patch("vid_summerizer.AnthropicGenerator.is_available", return_value=True):
            with patch("vid_summerizer.OpenAIGenerator.is_available", return_value=False):
                with patch("vid_summerizer.GeminiGenerator.is_available", return_value=False):
                    # Create the generator
                    generator = LLMSummaryGenerator(
                        output_dir="/tmp/test_output"
                    )
                    
                    # Mock _generate_scene_summary
                    with patch.object(generator, "_generate_scene_summary", return_value="Test summary"):
                        # Mock file writing
                        with patch("builtins.open", MagicMock()):
                            with patch("os.path.join", return_value="/tmp/test_output/summary_in_progress.md"):
                                result = generator.run([scene])
                                
                                # Verify the result includes proper formatting
                                assert "# Video Summary" in result
                                assert "## Scene 1 -" in result
                                assert "Test summary" in result


def load_test_scene():
    """Load or create a test scene for testing."""
    # Create a basic test scene
    scene = Scene(
        scene_id=1,
        start=0.0,
        end=10.0,
        screenshot=None,  # Mock generator will handle this
        transcript="This is a test transcript for the LLM generator tests."
    )
    return scene


if __name__ == "__main__":
    # Run the tests directly
    pytest.main(["-xvs", __file__])
"""Base classes for LLM providers."""
from abc import ABC, abstractmethod
import os


class LLMGenerator(ABC):
    """Abstract base class for LLM API generators."""

    @abstractmethod
    def generate_content(self, prompt: str, image_data: str = None) -> str:
        """Generate content using the LLM API."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM API is available for use."""
        pass
    
    @property
    def name(self) -> str:
        """Get the name of this LLM provider."""
        return self.__class__.__name__


def get_available_llm_providers():
    """
    Return a dictionary of available LLM providers based on environment variables.
    
    Returns:
        dict: A dictionary with provider names as keys and availability as values
    """
    return {
        "anthropic": os.getenv("ANTHROPIC_API_KEY") is not None,
        "openai": os.getenv("OPENAI_API_KEY") is not None,
        "gemini": os.getenv("GEMINI_API_KEY") is not None
    }
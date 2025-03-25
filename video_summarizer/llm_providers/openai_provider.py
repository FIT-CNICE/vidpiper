"""OpenAI API provider implementation."""
import os
from .base import LLMGenerator


class OpenAIGenerator(LLMGenerator):
    """Generator using OpenAI's API."""

    def __init__(self, model: str = "gpt-4o-2024-11-20",
                max_tokens: int = 2000):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The openai package is required. Install it with 'pip install openai'.")

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
        if self.api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

    def is_available(self) -> bool:
        return self.api_key is not None and self.client is not None

    def generate_content(self, prompt: str, image_data: str = None) -> str:
        if not self.is_available():
            raise ValueError("OpenAI API is not available.")

        system_message = (
            "You are a technical content summarizer specializing in creating accessible, "
            "accurate summaries of technical presentations and demos for general audiences. "
            "Use BOTH THE VISUAL CONTENT AND TRANSCRIPT to create comprehensive summaries. "
            "Transcript CONTAINS typo of technical jargons. "
            "For any technical terms or concepts, provide intuitive explanations that are "
            "accessible to non-experts while maintaining technical accuracy. "
            "Identify key technical concepts, data points, and main arguments from both the visuals "
            "and the transcript, explaining complex ideas using analogies or simplified examples when appropriate. "
            "Create logically connected summaries that flow naturally from previous scenes while ensuring "
            "technical content is understandable to a broad audience.")

        messages = [
            {"role": "system", "content": system_message}
        ]

        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content
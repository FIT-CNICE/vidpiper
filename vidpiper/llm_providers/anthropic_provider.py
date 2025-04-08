"""Anthropic Claude API provider implementation."""

import os
from .base import LLMGenerator


class AnthropicGenerator(LLMGenerator):
    """Generator using Anthropic's Claude API."""

    def __init__(
        self, model: str = "claude-3-7-sonnet-20250219", max_tokens: int = 2000
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required. Install it with 'pip install anthropic'."
            )

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        return self.api_key is not None and self.client is not None

    def generate_content(self, prompt: str, image_data: str = None) -> str:
        if not self.is_available():
            raise ValueError("Anthropic API is not available.")

        system_prompt = (
            "You are a technical content summarizer specializing in creating accessible, "
            "accurate summaries of technical presentations and demos for general audiences. "
            "Use BOTH THE VISUAL CONTENT AND TRANSCRIPT to create comprehensive summaries. "
            "Transcript CONTAINS typo of technical jargons. "
            "For any technical terms or concepts, provide intuitive explanations that are "
            "accessible to non-experts while maintaining technical accuracy. "
            "Identify key technical concepts, data points, and main arguments from both the visuals "
            "and the transcript, explaining complex ideas using analogies or simplified examples when appropriate. "
            "Create logically connected summaries that flow naturally from previous scenes while ensuring "
            "technical content is understandable to a broad audience."
        )

        if image_data:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=min(self.max_tokens, 3000),
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        else:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=min(self.max_tokens, 3000),
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

        return message.content[0].text

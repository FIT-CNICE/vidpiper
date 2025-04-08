"""Google's Gemini API provider implementation."""

import os
from .base import LLMGenerator


class GeminiGenerator(LLMGenerator):
    """Generator using Google's Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash", max_tokens: int = 2000):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "The google-generativeai package is required. Install it with 'pip install google-generativeai'."
            )

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.genai = None
        if self.api_key:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.genai = genai

    def is_available(self) -> bool:
        return self.api_key is not None and self.genai is not None

    def generate_content(self, prompt: str, image_data: str = None) -> str:
        if not self.is_available():
            raise ValueError("Gemini API is not available.")

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

        if image_data and self.model == "gemini-2.0-flash":
            import base64
            from PIL import Image
            import io

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            model = self.genai.GenerativeModel(self.model)
            response = model.generate_content([system_prompt, prompt, image])
        else:
            model = self.genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content([system_prompt, prompt])

        return response.text

"""Summary generation stage using LLM providers."""
import os
import base64
import time
import random
from typing import Dict, List, Any, Optional

from ..core.pipeline import PipelineStage
from ..core.data_classes import PipelineResult, Scene
from ..llm_providers import (
    LLMGenerator,
    AnthropicGenerator,
    OpenAIGenerator,
    GeminiGenerator,
    get_available_llm_providers
)


class LLMSummaryGenerator(PipelineStage):
    """
    Generates markdown summaries for video scenes using different LLM APIs.

    This implementation can switch between different LLM providers (Anthropic, OpenAI, Gemini)
    based on availability and preference, with fallback mechanisms.
    """

    def __init__(self, model: str = "gemini-2.0-flash",
                 max_tokens: int = 2000, output_dir: str = "output",
                 preferred_provider: str = "gemini"):
        self.max_tokens = max_tokens
        self.output_dir = output_dir  # Store the output directory
        self.summaries = {}  # Store individual scene summaries
        self.previous_summary_context = ""  # Track previous summary for continuity
        self.processed_scenes_count = 0  # Track number of processed scenes

        # Initialize all available LLM generators
        self.generators = {
            "gemini": GeminiGenerator(
                model,
                max_tokens),
            "anthropic": AnthropicGenerator(
                "claude-3-7-sonnet-20250219",
                max_tokens),
            "openai": OpenAIGenerator(
                "gpt-4o-2024-11-20",
                max_tokens)
                 }

        # Set the preferred provider
        self.preferred_provider = preferred_provider.lower()

        # Get available providers
        available_providers = get_available_llm_providers()
        self.available_generators = [
            provider for provider, available in available_providers.items()
            if available
        ]

        if not self.available_generators:
            raise ValueError(
                "No LLM API keys found. Set at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")

        print(
            f"Available LLM providers: {', '.join(self.available_generators)}")

        # Set the active generator based on preference and availability
        if self.preferred_provider in self.available_generators:
            self.active_provider = self.preferred_provider
        else:
            self.active_provider = self.available_generators[0]

        print(
            f"Using {self.active_provider.upper()} as the primary LLM provider")

    def run(self, data: PipelineResult) -> PipelineResult:
        scenes = data.scenes
        
        if not scenes:
            raise ValueError("No scenes to summarize")
        
        # Ensure output directory exists and is set in the result
        if not data.output_dir:
            data.output_dir = self.output_dir
            
        os.makedirs(data.output_dir, exist_ok=True)
        os.makedirs(os.path.join(data.output_dir, "screenshots"), exist_ok=True)
            
        print(f"Generating summaries with {self.active_provider.upper()} API for {len(scenes)} scenes...")
        complete_summary = "# Video Summary\n\n"

        for i, scene in enumerate(scenes):
            scene_id = scene.scene_id
            screenshot_path = scene.screenshot
            transcript = scene.transcript
            start_time = scene.start
            end_time = scene.end

            print(
                f"Processing scene {scene_id} ({start_time:.2f}s - {end_time:.2f}s) with {self.active_provider.upper()}...")

            # Format timestamp as MM:SS
            minutes, seconds = divmod(int(start_time), 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            # Generate section heading
            scene_heading = f"## Scene {scene_id} - [{timestamp}]\n\n"

            # Add screenshot reference with a relative path
            screenshot_filename = os.path.basename(screenshot_path)
            # Use relative path for screenshots in the markdown
            relative_screenshot_path = f"./screenshots/{screenshot_filename}"
            screenshot_ref = f"![Scene {scene_id} Screenshot]({relative_screenshot_path})\n\n"

            # Get summary for this scene using multimodal API with awareness of
            # previous content
            scene_summary = self._generate_scene_summary(
                scene_id, screenshot_path, transcript, start_time, end_time,
                i, len(scenes))

            # Add this scene to the complete summary
            complete_summary += scene_heading
            complete_summary += screenshot_ref
            complete_summary += scene_summary

            # Only add separator if not the last scene
            if i < len(scenes) - 1:
                complete_summary += "\n\n---\n\n"

            # Save progress incrementally
            self.summaries[scene_id] = scene_summary
            self.processed_scenes_count += 1

            # Update previous context for continuity (limit to last 2 summaries
            # to control token usage)
            self.previous_summary_context = scene_summary

            # Save the current state of the summary in the output directory
            in_progress_file = os.path.join(
                data.output_dir, "summary_in_progress.md")
            with open(in_progress_file, "w", encoding="utf-8") as f:
                f.write(complete_summary)

        # Save the final summary with the video name
        video_path = data.video_path
        if video_path:
            video_basename = os.path.basename(video_path)
            video_name = os.path.splitext(video_basename)[0]
            summary_file = os.path.join(data.output_dir, f"{video_name}_sum.md")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(complete_summary)
                
            # Add the summary file path to the result
            data.summary_file = summary_file

        # Add the complete summary to the result
        data.complete_summary = complete_summary
        
        # Add metadata about the summary generation
        data.metadata["summary_generation"] = {
            "llm_provider": self.active_provider,
            "available_providers": self.available_generators,
            "tokens_per_scene": self.max_tokens,
        }
        
        return data

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 for API request."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _generate_scene_summary(
            self, scene_id: int, screenshot_path: str, transcript: str,
            start_time: float, end_time: float, scene_index: int,
            total_scenes: int) -> str:
        """
        Generate a summary for a single scene using the selected LLM API
        with retry mechanism for handling API errors and fallback to alternative providers.
        """
        # Encode the screenshot as base64
        base64_image = self._encode_image(screenshot_path)

        # Create the user prompt with different strategies depending on
        # position
        if scene_index == 0:
            # First scene - establish the topic
            context_directive = (
                "As this is the first scene, " +
                "establish the main topic and context of the presentation. " +
                "If the image content does not look like part of a demo, " +
                "presentation, or panel discussion, tell me using EXACTLY " +
                "the following words: \"NO USEFUL CONTENT FOUND!\""
            )
        elif scene_index == total_scenes - 1:
            # Last scene - wrap up and connect to previous content
            context_directive = (
                "This is the final scene. Connect it with previous " +
                "content and provide closure on the topic. " +
                "Previous summary context:"
                f" \"{self.previous_summary_context}...\"")
        else:
            # Middle scene - maintain continuity
            context_directive = (
                "Connect this scene with the previous content for " +
                "a cohesive summary. Previous summary context:"
                f" \"{self.previous_summary_context}...\"")

        # Optimize transcript inclusion based on language
        if "你好" in transcript or "我们" in transcript or "这是" in transcript:
            # For Chinese content, use a shorter portion to reduce tokens
            transcript_preview = transcript
            transcript_note = "Note: This appears to be primarily Chinese." + \
                " Transcript might contain typos on technical jargons."
        else:
            transcript_preview = transcript
            transcript_note = ""

        # Create the user prompt for balanced analysis suitable for general
        # audience
        user_prompt = (
            f"This is frame {scene_id} from a technical presentation video (timestamp: {start_time:.2f}s to {end_time:.2f}s).\n\n"
            f"Transcript preview: {transcript_preview}\n"
            f"{transcript_note}\n\n"
            f"{context_directive}\n\n"
            "IMPORTANT: USE BOTH THE VISUAL CONTENT AND TRANSCRIPT to create a comprehensive summary.\n\n"
            "Please provide a detailed summary that includes:\n"
            "1. Key technical concepts from both the slide/demo and speaker's explanation\n"
            "2. Any relevant data, diagrams, or metrics with context from the transcript\n"
            "3. The main point of this segment, explaining technical terms in an intuitive yet rigorous way\n\n"
            "Make your summary accessible to a general audience by explaining technical terms using everyday language, analogies, or simplified examples while maintaining technical accuracy.")

        # Initialize providers list with active provider first, then others as
        # fallbacks
        providers_to_try = [self.active_provider] + [p
                                                     for p in self.available_generators if p != self.active_provider]

        # Retry parameters
        max_retries = 3
        retry_delay = 2  # Initial delay in seconds

        for provider in providers_to_try:
            # Skip unavailable providers
            if provider not in self.available_generators:
                continue

            retry_count = 0
            generator = self.generators[provider]

            while retry_count < max_retries:
                try:
                    print(
                        f"Attempting to generate summary for scene {scene_id} using {provider.upper()} (attempt {retry_count+1}/{max_retries})...")

                    # Generate content using the current provider
                    generated_text = generator.generate_content(
                        user_prompt, base64_image)

                    # Format headings consistently
                    generated_text = generated_text.replace("\n## ", "\n### ")
                    generated_text = generated_text.replace("\n# ", "\n## ")

                    # If we're not using the active provider, consider
                    # switching
                    if provider != self.active_provider:
                        print(
                            f"Successfully generated content with fallback provider {provider.upper()}. " +
                            f"Switching active provider from {self.active_provider.upper()} to {provider.upper()}")
                        self.active_provider = provider

                    print(
                        f"Generated summary for scene {scene_id} using {provider.upper()}")
                    return generated_text

                except Exception as e:
                    retry_count += 1
                    error_str = str(e)
                    error_msg = f"Error generating summary with {provider.upper()} for scene " + \
                        f"{scene_id} (attempt {retry_count}/{max_retries}): {error_str}"
                    print(error_msg)

                    # Check for overload or rate limit errors
                    is_overload = any(
                        phrase in error_str.lower()
                        for phrase
                        in
                        ["overloaded", "rate limit", "429", "quota",
                         "capacity"])

                    if retry_count < max_retries:
                        # Exponential backoff with jitter
                        # Add random jitter between 0-0.5 seconds
                        jitter = random.uniform(0, 0.5)
                        # Use longer backoff for overload errors
                        base_delay = retry_delay * 5 if is_overload else retry_delay
                        sleep_time = (
                            base_delay * (2 ** (retry_count - 1))) + jitter
                        print(
                            f"Retrying in {sleep_time:.2f} seconds..." +
                            (" (Rate limit/overload detected)" if is_overload else ""))
                        time.sleep(sleep_time)
                    else:
                        # All retries with current provider failed, try next
                        # provider
                        print(
                            f"All attempts with {provider.upper()} failed. Trying next provider...")
                        break

        # If we get here, all providers failed
        return f"*Error generating summary for scene {scene_id} after trying all available LLM providers*"

    def cleanup(self) -> None:
        """Clean up any resources used by this stage."""
        # Clean up the in-progress summary file if it exists
        in_progress_file = os.path.join(self.output_dir, "summary_in_progress.md")
        if os.path.exists(in_progress_file):
            try:
                os.remove(in_progress_file)
                print(f"Removed temporary file: {in_progress_file}")
            except Exception as e:
                print(f"Failed to remove temp file {in_progress_file}: {e}")


def create_summary_generator(
        model: str = "gemini-2.0-flash",
        max_tokens: int = 2000, 
        output_dir: str = "output",
        preferred_provider: str = "gemini") -> LLMSummaryGenerator:
    """Factory function to create a summary generator stage."""
    return LLMSummaryGenerator(
        model=model,
        max_tokens=max_tokens,
        output_dir=output_dir,
        preferred_provider=preferred_provider
    )
"""Summary updater stage for revising video summaries based on human feedback."""

import os
import time
import random
from typing import Optional

from ..core.pipeline import PipelineStage
from ..core.data_classes import PipelineResult
from ..llm_providers import (
    AnthropicGenerator,
    OpenAIGenerator,
    GeminiGenerator,
    get_available_llm_providers,
)


class SummaryUpdater(PipelineStage):
    """
    Updates generated summaries based on human feedback.

    This stage takes an existing summary file and human feedback comments,
    then generates an updated summary incorporating the feedback.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 2000,
        output_dir: Optional[str] = None,
        preferred_provider: str = "gemini",
    ):
        self.max_tokens = max_tokens
        self.output_dir = output_dir

        # Initialize all available LLM generators
        self.generators = {
            "gemini": GeminiGenerator(model, max_tokens),
            "anthropic": AnthropicGenerator(
                "claude-3-7-sonnet-20250219", max_tokens
            ),
            "openai": OpenAIGenerator("gpt-4o-2024-11-20", max_tokens),
        }

        # Set the preferred provider
        self.preferred_provider = preferred_provider.lower()

        # Get available providers
        available_providers = get_available_llm_providers()
        self.available_generators = [
            provider
            for provider, available in available_providers.items()
            if available
        ]

        if not self.available_generators:
            raise ValueError(
                "No LLM API keys found. Set at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
            )

        print(
            f"Available LLM providers: {', '.join(self.available_generators)}"
        )

        # Set the active generator based on preference and availability
        if self.preferred_provider in self.available_generators:
            self.active_provider = self.preferred_provider
        else:
            self.active_provider = self.available_generators[0]

        print(
            f"Using {self.active_provider.upper()} as the primary LLM provider for summary updates"
        )

    def run(self, data: PipelineResult) -> PipelineResult:
        """
        Run the summary updater stage.

        This method requires a summary_file in the data and a 'feedback' field in metadata.
        It will generate an updated summary incorporating the feedback.
        """
        # Determine output directory
        if not self.output_dir and data.output_dir:
            self.output_dir = data.output_dir
        elif not self.output_dir:
            self.output_dir = "output"

        os.makedirs(self.output_dir, exist_ok=True)

        # Check if we have a summary file to update
        if not data.summary_file or not os.path.exists(data.summary_file):
            raise ValueError("No summary file to update")

        # Check if we have feedback in the metadata
        feedback = data.metadata.get("feedback")
        if not feedback:
            raise ValueError("No feedback provided in metadata")

        # Read the original summary
        with open(data.summary_file, "r", encoding="utf-8") as f:
            original_summary = f.read()

        # Generate the updated summary
        updated_summary = self._update_summary(original_summary, feedback)

        # Save the updated summary
        base_name = os.path.basename(data.summary_file)
        name_without_ext = os.path.splitext(base_name)[0]

        # Create a new filename with '_updated' suffix
        if name_without_ext.endswith("_sum"):
            output_name = name_without_ext[:-4] + "_updated.md"
        else:
            output_name = name_without_ext + "_updated.md"

        output_path = os.path.join(self.output_dir, output_name)

        # Write the updated summary to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(updated_summary)

        print(f"Updated summary saved to: {output_path}")

        # Update the pipeline result
        data.metadata["updated_summary"] = output_path

        # Track both the original and updated summary
        data.metadata["summary_update"] = {
            "original_file": data.summary_file,
            "updated_file": output_path,
            "feedback_applied": feedback,
            "update_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return data

    def _update_summary(self, original_summary: str, feedback: str) -> str:
        """
        Update a summary based on human feedback.

        Args:
            original_summary (str): The original summary content
            feedback (str): Human feedback for improving the summary

        Returns:
            str: The updated summary incorporating the feedback
        """
        # Create prompt for the LLM
        prompt = (
            f"You are a professional technical writer tasked with updating a video summary based on human feedback.\n\n"
            f"Below is the original summary of a technical video, followed by human feedback comments. "
            f"Your task is to revise the summary to incorporate all the feedback while maintaining the "
            f"original structure and technical accuracy.\n\n"
            f"IMPORTANT GUIDELINES:\n"
            f"1. Address ALL points in the feedback completely\n"
            f"2. Maintain the original headings and section structure\n"
            f"3. Preserve technical accuracy while making the suggested improvements\n"
            f"4. Ensure the updated summary flows naturally and reads professionally\n"
            f"5. Maintain all existing image references and links\n"
            f"6. Keep the same level of technical detail as the original while incorporating feedback\n\n"
            f"ORIGINAL SUMMARY:\n\n{original_summary}\n\n"
            f"HUMAN FEEDBACK:\n\n{feedback}\n\n"
            f"Please provide the updated summary that incorporates all the feedback:"
        )

        # Initialize providers list with active provider first, then others as
        # fallbacks
        providers_to_try = [self.active_provider] + [
            p for p in self.available_generators if p != self.active_provider
        ]

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
                        f"Attempting to update summary using {provider.upper()} (attempt {retry_count + 1}/{max_retries})..."
                    )

                    # Generate updated content using the current provider
                    updated_text = generator.generate_content(prompt)

                    # If we're not using the active provider, consider
                    # switching
                    if provider != self.active_provider:
                        print(
                            f"Successfully generated content with fallback provider {provider.upper()}. "
                            + f"Switching active provider from {self.active_provider.upper()} to {provider.upper()}"
                        )
                        self.active_provider = provider

                    print(
                        f"Successfully updated summary using {provider.upper()}"
                    )
                    return updated_text

                except Exception as e:
                    retry_count += 1
                    error_str = str(e)
                    error_msg = (
                        f"Error updating summary with {provider.upper()} "
                        + f"(attempt {retry_count}/{max_retries}): {error_str}"
                    )
                    print(error_msg)

                    # Check for overload or rate limit errors
                    is_overload = any(
                        phrase in error_str.lower()
                        for phrase in [
                            "overloaded",
                            "rate limit",
                            "429",
                            "quota",
                            "capacity",
                        ]
                    )

                    if retry_count < max_retries:
                        # Exponential backoff with jitter
                        # Add random jitter between 0-0.5 seconds
                        jitter = random.uniform(0, 0.5)
                        # Use longer backoff for overload errors
                        base_delay = (
                            retry_delay * 5 if is_overload else retry_delay
                        )
                        sleep_time = (
                            base_delay * (2 ** (retry_count - 1))
                        ) + jitter
                        print(
                            f"Retrying in {sleep_time:.2f} seconds..."
                            + (
                                " (Rate limit/overload detected)"
                                if is_overload
                                else ""
                            )
                        )
                        time.sleep(sleep_time)
                    else:
                        # All retries with current provider failed, try next
                        # provider
                        print(
                            f"All attempts with {provider.upper()} failed. Trying next provider..."
                        )
                        break

        # If we get here, all providers failed
        error_msg = (
            "*Error updating summary after trying all available LLM providers*"
        )
        print(error_msg)

        # Return original summary with error message prepended
        return f"{error_msg}\n\n{original_summary}"

    def update_summary_file(
        self, summary_file: str, feedback: str, output_dir: Optional[str] = None
    ) -> str:
        """
        Update a specific summary file based on feedback.

        This is a standalone method that can be called without going through the pipeline.

        Args:
            summary_file (str): Path to the summary file to update
            feedback (str): Human feedback to incorporate
            output_dir (str, optional): Directory to save the updated summary

        Returns:
            str: Path to the updated summary file
        """
        # Use instance output_dir if none provided
        if not output_dir:
            output_dir = self.output_dir or os.path.dirname(summary_file)

        os.makedirs(output_dir, exist_ok=True)

        # Read the original summary
        with open(summary_file, "r", encoding="utf-8") as f:
            original_summary = f.read()

        # Generate the updated summary
        updated_summary = self._update_summary(original_summary, feedback)

        # Create the output filename
        base_name = os.path.basename(summary_file)
        name_without_ext = os.path.splitext(base_name)[0]

        # Create a new filename with '_updated' suffix
        if name_without_ext.endswith("_sum"):
            output_name = name_without_ext[:-4] + "_updated.md"
        else:
            output_name = name_without_ext + "_updated.md"

        output_path = os.path.join(output_dir, output_name)

        # Write the updated summary
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(updated_summary)

        print(f"Updated summary saved to: {output_path}")
        return output_path

    def cleanup(self) -> None:
        """Clean up any resources used by this stage."""
        # Nothing to clean up for this stage
        pass


def create_summary_updater(
    model: str = "gemini-2.0-flash",
    max_tokens: int = 2000,
    output_dir: Optional[str] = None,
    preferred_provider: str = "gemini",
) -> SummaryUpdater:
    """Factory function to create a summary updater stage."""
    return SummaryUpdater(
        model=model,
        max_tokens=max_tokens,
        output_dir=output_dir,
        preferred_provider=preferred_provider,
    )

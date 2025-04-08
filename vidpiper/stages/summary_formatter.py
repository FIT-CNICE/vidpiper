"""Summary formatter stage for converting raw summaries into Marp slide decks."""

import os
import glob
from typing import List, Optional

from ..core.pipeline import PipelineStage
from ..core.data_classes import PipelineResult
from ..llm_providers import (
    AnthropicGenerator,
    OpenAIGenerator,
    GeminiGenerator,
    get_available_llm_providers,
)


class SummaryFormatter(PipelineStage):
    """
    Formats raw summaries into Marp slide decks.

    This stage takes raw summary files with "_sum.md" suffix and formats them
    into Marp slide decks with "_fmt.md" suffix. It can process individual files
    or entire directories of summary files.
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
            f"Using {self.active_provider.upper()} as the primary LLM provider for formatting"
        )

    def run(self, data: PipelineResult) -> PipelineResult:
        """
        Run the summary formatter stage.

        If a summary_file is specified in the data, format that file.
        If a directory is specified, find and format all "*_sum.md" files in it.
        """
        # Determine output directory
        if not self.output_dir and data.output_dir:
            self.output_dir = data.output_dir
        elif not self.output_dir:
            self.output_dir = "output"

        os.makedirs(self.output_dir, exist_ok=True)

        # If we have a specific summary file in the data, process it
        if data.summary_file and os.path.exists(data.summary_file):
            formatted_file = self._process_summary_file(data.summary_file)
            data.formatted_file = formatted_file
            data.metadata["formatted_summary"] = formatted_file
            return data

        # If we have a video path, process all summary files in its directory
        if data.video_path:
            video_dir = os.path.dirname(data.video_path)
            summary_files = self._find_summary_files(video_dir)

            if summary_files:
                formatted_files = []
                for summary_file in summary_files:
                    formatted_file = self._process_summary_file(summary_file)
                    formatted_files.append(formatted_file)

                data.metadata["formatted_summaries"] = formatted_files

        return data

    def _find_summary_files(self, directory: str) -> List[str]:
        """Find all summary files with "_sum.md" suffix in the given directory and its subdirectories."""
        pattern = os.path.join(directory, "**", "*_sum.md")
        return glob.glob(pattern, recursive=True)

    def _process_summary_file(self, summary_file: str) -> str:
        """Process a single summary file and convert it to a Marp slide deck."""
        print(f"Processing summary file: {summary_file}")

        # Read the summary file
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_content = f.read()

        # Generate the formatted content
        formatted_content = self._generate_marp_deck(
            summary_content, summary_file
        )

        # Determine the output file path
        base_name = os.path.basename(summary_file)
        name_without_ext = os.path.splitext(base_name)[0]
        # Remove '_sum' suffix if present and add '_fmt'
        if name_without_ext.endswith("_sum"):
            output_name = name_without_ext[:-4] + "_fmt.md"
        else:
            output_name = name_without_ext + "_fmt.md"

        output_path = os.path.join(self.output_dir, output_name)

        # Write the formatted content to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        print(f"Formatted summary saved to: {output_path}")
        return output_path

    def _generate_marp_deck(
        self, summary_content: str, summary_file: str
    ) -> str:
        """Generate a Marp slide deck from the summary content using LLM."""
        # Get the base directory for resolving relative paths
        base_dir = os.path.dirname(os.path.abspath(summary_file))

        # Expand relative links to absolute paths
        summary_content = self._expand_links(summary_content, base_dir)

        # Create prompt for the LLM
        prompt = (
            f"You are a specialized document formatter that creates professional Marp slide decks from technical content.\n\n"
            f"Your task is to convert the following raw technical summary into a professional Marp presentation slide deck.\n\n"
            f"IMPORTANT FORMATTING REQUIREMENTS:\n"
            f'1. Start with a YAML front matter(e.g. "---\nfield1: val1\nfield2: val2\n---\n")'
            "that includes marp-theme, title, subtitle, and taxonomy fields.\n"
            f"   - The marp-theme field has only 2 options: proposal or update"
            f'   - The taxonomy field should be a hierarchical categorization from "stem" to "leaf" based on the content.\n'
            f"   - The taxonomy field should be made of 3-6 keywords"
            f'   - Example: taxonomy: "Interconnect > Cable > Optical cable"\n'
            f"   - Wrap field value with double quotes if it contains non-alphabetic characters\n"
            f"2. Organize the content into self-contained & comprehensive points "
            "on each slide WITHOUT using Marp directives.\n"
            f'3. Use ONLY "---" for page breaks, and ONLY use SECOND-level tiltle for each slide\n'
            f"4. Preserve ALL technical details while making the presentation intuitive.\n"
            f"5. The last slide MUST be DETAIL-ORIENTED takeaways in points of complete and self-contained sentences\n"
            f"6. Mark the beginning of your formatted content with '---BEGIN MARP DECK---' and the ending with '---END MARP DECK---'.\n\n"
            f"7. NEVER, NEVER, NEVER generate front page(i.e. <!-- _class: front-page -->)!!!\n"
            f'8. use "![width:500px](url/to/img)" to set image width to 500px\n'
            f"9. DO NOT USE HTML COMMENTS for page breaks\n"
            f"Here is the content to format:\n\n{summary_content}"
        )

        # Use the active provider to generate the formatted content
        generator = self.generators[self.active_provider]
        try:
            formatted_text = generator.generate_content(prompt)
            formatted_text = formatted_text.replace("```yaml", "---")
            formatted_text = formatted_text.replace("---\n---", "---")
            formatted_text = formatted_text.replace(
                "<!-- _class: front-page -->\n---", "\n"
            )

            # Extract content between the markers
            if (
                "---BEGIN MARP DECK---" in formatted_text
                and "---END MARP DECK---" in formatted_text
            ):
                start_idx = formatted_text.find("---BEGIN MARP DECK---") + len(
                    "---BEGIN MARP DECK---"
                )
                end_idx = formatted_text.find("---END MARP DECK---")
                formatted_text = formatted_text[start_idx : end_idx - 1].strip()

            if not formatted_text.startswith("---\n"):
                formatted_text = "---\n" + formatted_text

            return formatted_text

        except Exception as e:
            print(
                f"Error generating Marp deck with {self.active_provider.upper()}: {str(e)}"
            )
            # Fallback message in case of error
            return (
                "---\n"
                "marp: true\n"
                f"title: Formatted Summary of {os.path.basename(summary_file)}\n"
                "author: Auto-generated\n"
                "theme: default\n"
                "---\n\n"
                "# Error Formatting Summary\n\n"
                f"There was an error formatting the summary using {self.active_provider.upper()}:\n\n"
                f"{str(e)}\n\n"
                "---\n\n"
                "# Original Content\n\n"
                f"{summary_content}"
            )

    def _expand_links(self, content: str, base_dir: str) -> str:
        """Expand relative links in the content to absolute paths."""
        lines = content.split("\n")
        result_lines = []

        for line in lines:
            # Look for markdown image or link syntax
            import re

            img_links = re.findall(r"!\[(.*?)\]\((.*?)\)", line)
            text_links = re.findall(r"(?<!!)\[(.*?)\]\((.*?)\)", line)

            # Process all found links
            for alt_text, link in img_links + text_links:
                # Only process relative links (not http/https)
                if not link.startswith(("http://", "https://")):
                    # Convert to absolute path if it's relative
                    abs_path = os.path.abspath(os.path.join(base_dir, link))
                    # Replace the relative link with the absolute one
                    if os.path.exists(abs_path):
                        line = line.replace(f"]({link})", f"]({abs_path})")

            result_lines.append(line)

        return "\n".join(result_lines)

    def format_directory(self, directory: str) -> List[str]:
        """
        Format all summary files in the given directory and its subdirectories.

        This is a standalone method that can be called without going through the pipeline.
        """
        summary_files = self._find_summary_files(directory)
        formatted_files = []

        for summary_file in summary_files:
            formatted_file = self._process_summary_file(summary_file)
            formatted_files.append(formatted_file)

        return formatted_files

    def cleanup(self) -> None:
        """Clean up any resources used by this stage."""
        # Nothing to clean up for this stage
        pass


def create_summary_formatter(
    model: str = "gemini-2.0-flash",
    max_tokens: int = 2000,
    output_dir: Optional[str] = None,
    preferred_provider: str = "gemini",
) -> SummaryFormatter:
    """Factory function to create a summary formatter stage."""
    return SummaryFormatter(
        model=model,
        max_tokens=max_tokens,
        output_dir=output_dir,
        preferred_provider=preferred_provider,
    )

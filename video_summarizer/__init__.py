# Video Summarizer Package
# Main pipeline and components for video summarization

from .core import Scene, PipelineResult, Pipeline, PipelineStage, CustomStage
from .stages import (
    SceneDetector,
    SceneProcessor,
    LLMSummaryGenerator,
    create_scene_detector,
    create_scene_processor,
    create_summary_generator,
)
from .llm_providers import (
    LLMGenerator,
    AnthropicGenerator,
    OpenAIGenerator,
    GeminiGenerator,
    get_available_llm_providers,
)

__all__ = [
    # Core
    "Scene",
    "PipelineResult",
    "Pipeline",
    "PipelineStage",
    "CustomStage",
    # Stages
    "SceneDetector",
    "SceneProcessor",
    "LLMSummaryGenerator",
    "create_scene_detector",
    "create_scene_processor",
    "create_summary_generator",
    # LLM Providers
    "LLMGenerator",
    "AnthropicGenerator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "get_available_llm_providers",
]

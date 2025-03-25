# Core components for the video summarizer pipeline
from .data_classes import Scene, PipelineResult
from .pipeline import Pipeline, PipelineStage, CustomStage

__all__ = [
    'Scene',
    'PipelineResult',
    'Pipeline',
    'PipelineStage',
    'CustomStage'
]
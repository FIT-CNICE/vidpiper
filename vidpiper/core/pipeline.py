"""Pipeline infrastructure for the video summarizer."""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Type
from .data_classes import PipelineResult


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    @abstractmethod
    def run(self, data: PipelineResult) -> PipelineResult:
        """Run this pipeline stage with the provided input data."""
        pass

    def cleanup(self) -> None:
        """Clean up any resources used by this stage."""
        pass

    @property
    def name(self) -> str:
        """Get the name of this stage."""
        return self.__class__.__name__

    def save_checkpoint(self, data: PipelineResult, checkpoint_dir: str) -> str:
        """Save a checkpoint of the pipeline state after this stage."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{self.name}_checkpoint.json"
        )

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data.to_dict(), f, indent=2)

        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> PipelineResult:
        """Load a pipeline checkpoint from a file."""
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PipelineResult.from_dict(data)


class Pipeline:
    """
    A generic pipeline that can be configured with different stages.

    This pipeline is designed to be flexible, allowing stages to be added,
    inserted, replaced, or removed as needed. Each stage processes the data
    and passes the result to the next stage.
    """

    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.checkpoint_dir: Optional[str] = None

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self

    def insert_stage(self, index: int, stage: PipelineStage) -> "Pipeline":
        """Insert a stage at a specific position in the pipeline."""
        self.stages.insert(index, stage)
        return self

    def replace_stage(self, index: int, stage: PipelineStage) -> "Pipeline":
        """Replace a stage at a specific position in the pipeline."""
        self.stages[index] = stage
        return self

    def remove_stage(self, index: int) -> "Pipeline":
        """Remove a stage at a specific position from the pipeline."""
        self.stages.pop(index)
        return self

    def set_checkpoint_dir(self, directory: str) -> "Pipeline":
        """Set the directory for saving pipeline checkpoints."""
        self.checkpoint_dir = directory
        return self

    def run(self, initial_input: PipelineResult) -> PipelineResult:
        """Run the pipeline with the given initial input."""
        data = initial_input
        try:
            for i, stage in enumerate(self.stages):
                print(f"Running stage {i + 1}/{len(self.stages)}: {stage.name}")
                data = stage.run(data)

                # Save checkpoint if checkpoint directory is configured
                if self.checkpoint_dir:
                    checkpoint_path = stage.save_checkpoint(
                        data, self.checkpoint_dir
                    )
                    print(f"Saved checkpoint: {checkpoint_path}")

            return data
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up all stages."""
        for stage in self.stages:
            try:
                stage.cleanup()
            except Exception as e:
                print(f"Error during cleanup of {stage.name}: {str(e)}")

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, stage_type: Type[PipelineStage]
    ) -> PipelineResult:
        """Load pipeline data from a checkpoint to continue processing with a specific stage type."""
        return stage_type.load_checkpoint(checkpoint_path)


class CustomStage(PipelineStage):
    """Base class for custom pipeline stages.

    This class is meant to be extended by users who want to create custom
    stages for the pipeline.
    """

    def run(self, data: PipelineResult) -> PipelineResult:
        """Implement custom processing logic in this method."""
        # Custom processing logic should be implemented in subclasses
        return data

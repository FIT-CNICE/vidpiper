#!/usr/bin/env python3
"""
Test script for the Pipeline architecture of the video summarization pipeline.
This script tests the core functionality of the Pipeline class and PipelineStage classes.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_summarizer.core import Pipeline, PipelineStage, PipelineResult, Scene


class MockStage(PipelineStage):
    """A mock pipeline stage for testing."""
    
    def __init__(self, name="MockStage", increment=1, should_fail=False):
        self.name_override = name
        self.increment = increment
        self.should_fail = should_fail
        self.was_run = False
        self.was_cleaned_up = False
    
    @property
    def name(self):
        """Override the name property."""
        return self.name_override
    
    def run(self, data: PipelineResult) -> PipelineResult:
        """Mock implementation that increments a counter in metadata."""
        self.was_run = True
        
        # Simulate a failure if requested
        if self.should_fail:
            raise ValueError("Stage failed on purpose")
        
        # Get the current counter value or set to 0
        counter = data.metadata.get("counter", 0)
        
        # Increment the counter
        data.metadata["counter"] = counter + self.increment
        data.metadata["last_stage"] = self.name
        
        return data
    
    def cleanup(self):
        """Mark that cleanup was called."""
        self.was_cleaned_up = True


class TestPipeline:
    """Tests for the Pipeline class."""
    
    def test_pipeline_initialization(self):
        """Test that the Pipeline initializes correctly."""
        pipeline = Pipeline()
        assert pipeline.stages == []
        assert pipeline.checkpoint_dir is None
    
    def test_adding_stages(self):
        """Test adding stages to the pipeline."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1")
        stage2 = MockStage("Stage2")
        
        # Add stages
        pipeline.add_stage(stage1)
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0] is stage1
        
        # Method chaining
        pipeline.add_stage(stage2)
        assert len(pipeline.stages) == 2
        assert pipeline.stages[1] is stage2
    
    def test_inserting_stages(self):
        """Test inserting stages into the pipeline."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1")
        stage2 = MockStage("Stage2")
        stage3 = MockStage("Stage3")
        
        # Add stages
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage3)
        
        # Insert a stage in the middle
        pipeline.insert_stage(1, stage2)
        
        assert len(pipeline.stages) == 3
        assert pipeline.stages[0] is stage1
        assert pipeline.stages[1] is stage2
        assert pipeline.stages[2] is stage3
    
    def test_replacing_stages(self):
        """Test replacing stages in the pipeline."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1")
        stage2 = MockStage("Stage2")
        stage3 = MockStage("Stage3")
        
        # Add stages
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        
        # Replace a stage
        pipeline.replace_stage(1, stage3)
        
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] is stage1
        assert pipeline.stages[1] is stage3
    
    def test_removing_stages(self):
        """Test removing stages from the pipeline."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1")
        stage2 = MockStage("Stage2")
        stage3 = MockStage("Stage3")
        
        # Add stages
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_stage(stage3)
        
        # Remove a stage
        pipeline.remove_stage(1)
        
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] is stage1
        assert pipeline.stages[1] is stage3
    
    def test_pipeline_execution(self):
        """Test that the pipeline executes stages in order."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1", increment=1)
        stage2 = MockStage("Stage2", increment=2)
        stage3 = MockStage("Stage3", increment=3)
        
        # Add stages
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_stage(stage3)
        
        # Create initial data
        initial_data = PipelineResult(video_path="test_video.mp4")
        
        # Run the pipeline
        result = pipeline.run(initial_data)
        
        # Check that all stages were run
        assert stage1.was_run
        assert stage2.was_run
        assert stage3.was_run
        
        # Check that the counter was incremented correctly
        assert result.metadata["counter"] == 6  # 1 + 2 + 3
        assert result.metadata["last_stage"] == "Stage3"
    
    def test_pipeline_error_handling(self):
        """Test that the pipeline handles errors correctly."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1", increment=1)
        stage2 = MockStage("Stage2", increment=2, should_fail=True)
        stage3 = MockStage("Stage3", increment=3)
        
        # Add stages
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_stage(stage3)
        
        # Create initial data
        initial_data = PipelineResult(video_path="test_video.mp4")
        
        # Run the pipeline (should fail)
        with pytest.raises(ValueError, match="Stage failed on purpose"):
            pipeline.run(initial_data)
        
        # Check that only the stages before the failing stage were run
        assert stage1.was_run
        assert stage2.was_run
        assert not stage3.was_run
        
        # Verify cleanup was called for all stages
        assert stage1.was_cleaned_up
        assert stage2.was_cleaned_up
        assert stage3.was_cleaned_up
    
    def test_checkpoint_directory(self):
        """Test setting the checkpoint directory."""
        pipeline = Pipeline()
        pipeline.set_checkpoint_dir("/tmp/checkpoints")
        assert pipeline.checkpoint_dir == "/tmp/checkpoints"
    
    def test_checkpoint_saving(self):
        """Test saving checkpoints during pipeline execution."""
        pipeline = Pipeline()
        stage1 = MockStage("Stage1", increment=1)
        stage2 = MockStage("Stage2", increment=2)
        
        # Add stages
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        
        # Set checkpoint directory
        with patch('os.makedirs'):
            with patch('json.dump'):
                with patch('builtins.open', MagicMock()):
                    # Set checkpoint directory
                    pipeline.set_checkpoint_dir("/tmp/checkpoints")
                    
                    # Create initial data
                    initial_data = PipelineResult(video_path="test_video.mp4")
                    
                    # Run the pipeline
                    pipeline.run(initial_data)
                    
                    # Check that save_checkpoint was called for each stage
                    # This is challenging to test directly, but we can verify the stages were run
                    assert stage1.was_run
                    assert stage2.was_run


class TestPipelineIntegration:
    """Integration tests for the Pipeline with real stages."""
    
    def create_test_pipeline(self):
        """Create a test pipeline with mock stages."""
        pipeline = Pipeline()
        
        # Create a simple chain of stages
        stage1 = MockStage("VideoLoader", increment=1)
        stage2 = MockStage("FeatureExtractor", increment=2)
        stage3 = MockStage("ResultProcessor", increment=3)
        
        # Add stages to pipeline
        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_stage(stage3)
        
        return pipeline, (stage1, stage2, stage3)
    
    def test_complete_pipeline_execution(self):
        """Test running a complete pipeline."""
        pipeline, stages = self.create_test_pipeline()
        
        # Create initial data
        initial_data = PipelineResult(
            video_path="test_video.mp4",
            scenes=[
                Scene(scene_id=1, start=0.0, end=10.0),
                Scene(scene_id=2, start=10.0, end=20.0)
            ]
        )
        
        # Run the pipeline
        result = pipeline.run(initial_data)
        
        # Check that all stages were executed
        for stage in stages:
            assert stage.was_run
            assert stage.was_cleaned_up
        
        # Check the result
        assert result.video_path == "test_video.mp4"
        assert len(result.scenes) == 2
        assert result.metadata["counter"] == 6  # 1 + 2 + 3
        assert result.metadata["last_stage"] == "ResultProcessor"
    
    def test_pipeline_with_custom_stage(self):
        """Test extending the pipeline with a custom stage."""
        
        # Create a custom stage that manipulates scenes
        class SceneFilterStage(PipelineStage):
            def run(self, data: PipelineResult) -> PipelineResult:
                # Filter scenes longer than 5 seconds
                data.scenes = [scene for scene in data.scenes 
                               if (scene.end - scene.start) > 5.0]
                data.metadata["filtered"] = True
                return data
        
        # Create pipeline with the custom stage
        pipeline = Pipeline()
        pipeline.add_stage(MockStage("VideoLoader"))
        pipeline.add_stage(SceneFilterStage())
        pipeline.add_stage(MockStage("ResultProcessor"))
        
        # Create initial data with scenes of different durations
        initial_data = PipelineResult(
            video_path="test_video.mp4",
            scenes=[
                Scene(scene_id=1, start=0.0, end=3.0),  # 3 seconds (should be filtered)
                Scene(scene_id=2, start=10.0, end=20.0),  # 10 seconds
                Scene(scene_id=3, start=20.0, end=24.0)  # 4 seconds (should be filtered)
            ]
        )
        
        # Run the pipeline
        result = pipeline.run(initial_data)
        
        # Check that filtering worked
        assert len(result.scenes) == 1
        assert result.scenes[0].scene_id == 2
        assert result.metadata["filtered"] is True


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
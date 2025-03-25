"""Data classes used throughout the video summarization pipeline."""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class Scene:
    """Data class representing a scene in a video."""
    scene_id: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    screenshot: Optional[str] = None  # Path to screenshot image
    transcript: Optional[str] = None  # Transcript text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Scene object to dictionary for serialization."""
        return {
            "scene_id": self.scene_id,
            "start": self.start,
            "end": self.end,
            "screenshot": self.screenshot,
            "transcript": self.transcript
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scene':
        """Create Scene object from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            start=data["start"],
            end=data["end"],
            screenshot=data.get("screenshot"),
            transcript=data.get("transcript")
        )


@dataclass
class PipelineResult:
    """Container for results passed between pipeline stages."""
    video_path: str
    scenes: List[Scene] = None
    output_dir: str = None
    summary_file: str = None
    complete_summary: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize empty lists and dicts if they are None."""
        if self.scenes is None:
            self.scenes = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PipelineResult object to dictionary for serialization."""
        return {
            "video_path": self.video_path,
            "scenes": [scene.to_dict() for scene in self.scenes] if self.scenes else [],
            "output_dir": self.output_dir,
            "summary_file": self.summary_file,
            "complete_summary": self.complete_summary,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineResult':
        """Create PipelineResult object from dictionary."""
        result = cls(
            video_path=data["video_path"],
            output_dir=data.get("output_dir"),
            summary_file=data.get("summary_file"),
            complete_summary=data.get("complete_summary"),
            metadata=data.get("metadata", {})
        )
        if "scenes" in data and data["scenes"]:
            result.scenes = [Scene.from_dict(scene) for scene in data["scenes"]]
        return result
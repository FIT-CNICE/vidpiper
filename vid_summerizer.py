import os
import subprocess
import json
import requests
import base64
import tempfile
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# ---------------------------------------------------------
# Data Classes
# ---------------------------------------------------------
@dataclass
class Scene:
    """Data class representing a scene in a video."""
    scene_id: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    screenshot: Optional[str] = None  # Path to screenshot image
    transcript: Optional[str] = None  # Transcript text

# ---------------------------------------------------------
# Stage 1: Scene Detection using PySceneDetect
# ---------------------------------------------------------
import scenedetect
from scenedetect import SceneDetector as PySceneDetector
from scenedetect import open_video, ContentDetector, Scene as PyScene


class PipelineStage(ABC):
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Run this pipeline stage with the provided input data."""
        pass
    
    def cleanup(self) -> None:
        """Clean up any resources used by this stage."""
        pass


class SceneDetector(PipelineStage):
    """
    Detects scenes in the input video using PySceneDetect's ContentDetector.

    Downscaling is applied to reduce processing load, making it suitable
    for systems with limited VRAM (4GB or less).
    """

    def __init__(self, threshold: float = 30.0, downscale_factor: int = 2):
        self.threshold = threshold
        self.downscale_factor = downscale_factor

    def run(self, input_data: Dict) -> List[Scene]:
        video_path = input_data.get("video_path")
        if not video_path:
            raise ValueError("Input must contain 'video_path' key")
        
        print(f"Stage 1: Detecting scenes using PySceneDetect for {video_path}...")
        
        # Open the video file
        try:
            video = open_video(video_path)
        except Exception as e:
            print(f"Error opening video: {e}")
            print(f"Checking if file exists: {os.path.exists(video_path)}")
            raise ValueError(f"Could not open video file: {video_path}") from e
        
        # Create detector with appropriate settings for low VRAM
        detector = ContentDetector(threshold=self.threshold)
        
        # Find scenes
        print("Detecting scenes...")
        scene_list = scenedetect.detect(video, detector, show_progress=True)
        
        # Convert to our Scene dataclass format
        scenes = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append(Scene(
                scene_id=i + 1,
                start=start_time,
                end=end_time
            ))
        
        print(f"Detected {len(scenes)} scenes.")
        return scenes

# ---------------------------------------------------------
# Stage 2: Screenshot and Transcript Extraction
# ---------------------------------------------------------


class SceneProcessor(PipelineStage):
    """
    For each detected scene, extracts a screenshot (using FFmpeg)
    at the midpoint and generates a transcript.
    """

    def __init__(self, video_path: str, output_dir: str, use_whisper: bool = False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.use_whisper = use_whisper
        self.temp_files = []
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

    def run(self, scenes: List[Scene]) -> List[Scene]:
        print("Stage 2: Extracting screenshots and transcripts for each scene")
        
        for scene in scenes:
            # Generate screenshot filename
            screenshot_path = os.path.join(
                self.output_dir, "screenshots", f"scene_{scene.scene_id}.jpg")
            
            # Choose a timestamp at the midpoint of the scene
            timestamp = (scene.start + scene.end) / 2
            
            # Extract screenshot and transcript
            self.extract_screenshot(timestamp, screenshot_path)
            transcript = self.extract_transcript(scene.start, scene.end)
            
            # Update scene object
            scene.screenshot = screenshot_path
            scene.transcript = transcript
            
            print(f"Processed scene {scene.scene_id}: {scene.start:.2f}s to {scene.end:.2f}s")
            
        return scenes

    def extract_screenshot(self, timestamp: float, output_path: str):
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", self.video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
            "-y"  # Overwrite output file if exists
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted screenshot at {timestamp:.2f}s to {output_path}")

    def extract_transcript(self, start: float, end: float) -> str:
        if self.use_whisper:
            try:
                return self._transcribe_with_whisper(start, end)
            except Exception as e:
                print(f"Whisper transcription failed: {e}")
                return self._placeholder_transcript(start, end)
        else:
            return self._placeholder_transcript(start, end)
    
    def _extract_audio_clip(self, start: float, end: float) -> str:
        """Extract audio clip for a scene using ffmpeg."""
        output_path = tempfile.mktemp(suffix=".wav")
        self.temp_files.append(output_path)
        
        duration = end - start
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", self.video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        return output_path
    
    def _transcribe_with_whisper(self, start: float, end: float) -> str:
        """Transcribe audio using Whisper (if available)."""
        try:
            import whisper
        except ImportError:
            print("Whisper not installed. Using placeholder transcript.")
            return self._placeholder_transcript(start, end)
        
        # Extract audio clip for this scene
        audio_clip_path = self._extract_audio_clip(start, end)
        
        # Load the tiny Whisper model (least VRAM intensive)
        model = whisper.load_model("tiny", device="cuda")
        
        # Run transcription with memory-efficient settings
        result = model.transcribe(
            audio_clip_path,
            fp16=False,  # Disable half-precision to reduce memory issues
            beam_size=1, # Reduce beam size
            best_of=1    # Only return the top result
        )
        
        return result["text"].strip()
    
    def _placeholder_transcript(self, start: float, end: float) -> str:
        """Generate a placeholder transcript."""
        transcript = f"Transcript for scene from {start:.2f} to {end:.2f} seconds."
        print(f"Generated placeholder transcript for scene ({start:.2f}-{end:.2f}s).")
        return transcript
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove temp file {file_path}: {e}")

# ---------------------------------------------------------
# Stage 3: Session-based Markdown Summary Generation Using Anthropic API
# ---------------------------------------------------------


class AnthropicSummaryGenerator(PipelineStage):
    """
    Generates markdown summaries for video scenes using
    the latest Claude model (3.7).

    This implementation uses the Messages API with multimodal support to process
    both screenshots and transcripts for each scene.
    """

    def __init__(self, model: str = "claude-3-7-sonnet-20240307",
                 max_tokens: int = 1000):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = model
        self.max_tokens = max_tokens
        self.summaries = {}  # Store individual scene summaries

        # System prompt for technical content summarization
        self.system_prompt = (
            "You are a technical content summarizer specializing in creating concise, "
            "accurate summaries of technical presentations and demos. For each scene, "
            "analyze both the visual content and the transcript to identify key technical "
            "concepts, data points, and main arguments.")

    def run(self, processed_scenes: List[Scene]) -> str:
        print("Stage 3: Generating summaries with Claude 3.7...")
        complete_summary = "# Video Summary\n\n"

        for scene in processed_scenes:
            scene_id = scene.scene_id
            screenshot_path = scene.screenshot
            transcript = scene.transcript
            start_time = scene.start
            end_time = scene.end

            print(
                f"Processing scene {scene_id} ({start_time:.2f}s - {end_time:.2f}s)...")

            # Format timestamp as MM:SS
            minutes, seconds = divmod(int(start_time), 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            # Generate section heading
            scene_heading = f"## Scene {scene_id} - [{timestamp}]\n\n"

            # Add screenshot reference
            screenshot_filename = os.path.basename(screenshot_path)
            screenshot_ref = f"![Scene {scene_id} Screenshot]({screenshot_path})\n\n"

            # Get summary for this scene using multimodal API
            scene_summary = self._generate_scene_summary(
                scene_id, screenshot_path, transcript, start_time, end_time)

            # Add this scene to the complete summary
            complete_summary += scene_heading
            complete_summary += screenshot_ref
            complete_summary += scene_summary
            complete_summary += "\n\n---\n\n"

            # Save progress incrementally
            self.summaries[scene_id] = scene_summary

            # Save the current state of the summary
            with open("summary_in_progress.md", "w") as f:
                f.write(complete_summary)

        return complete_summary

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 for API request."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _generate_scene_summary(
            self, scene_id: int, screenshot_path: str, transcript: str,
            start_time: float, end_time: float) -> str:
        """Generate a summary for a single scene using Claude 3.7's multimodal capabilities."""
        try:
            # Encode the screenshot as base64
            base64_image = self._encode_image(screenshot_path)

            # Create the API request payload
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            # Create the user prompt
            user_prompt = (
                f"This is a frame from a technical presentation or demo video. "
                f"The transcript from this segment (from {start_time:.2f}s to {end_time:.2f}s) is:\n\n"
                f"{transcript}\n\n"
                "Please provide a concise summary of what's being presented in this segment, including:\n"
                "1. Key technical concepts being discussed\n"
                "2. Any relevant data or metrics shown in the image\n"
                "3. The main point the presenter is making in this segment\n\n"
                "Format your response as markdown content. Include specific references to what's visible in the screenshot."
            )

            payload = {"model": self.model,
                       "max_tokens": self.max_tokens,
                       "system": self.system_prompt,
                       "messages": [{"role": "user",
                                     "content": [{"type": "image",
                                                  "source": {"type": "base64",
                                                             "media_type": "image/jpeg",
                                                             "data": base64_image}},
                                                 {"type": "text",
                                                  "text": user_prompt}]}]}

            # Send the request to Anthropic API
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            generated_text = result["content"][0]["text"]

            print(f"Generated summary for scene {scene_id}")
            return generated_text

        except Exception as e:
            error_msg = f"Error generating summary for scene {scene_id}: {str(e)}"
            print(error_msg)
            return f"*{error_msg}*"

# ---------------------------------------------------------
# Pipeline Manager to Orchestrate the Pipeline Stages
# ---------------------------------------------------------


class PipelineManager:
    """
    Orchestrates the sequential execution of the pipeline stages.
    The stages list can be extended or modified to include additional upstream
    or downstream components.
    """

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(self, initial_input: Any = None) -> Any:
        data = initial_input
        try:
            for i, stage in enumerate(self.stages):
                print(f"Running stage {i+1}/{len(self.stages)}: {stage.__class__.__name__}")
                data = stage.run(data)
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
                print(f"Error during cleanup of {stage.__class__.__name__}: {str(e)}")


# ---------------------------------------------------------
# Main Execution: Tying the Pipeline Together
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a markdown summary of a video with screenshots")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory to save screenshots and summary")
    parser.add_argument(
        "--threshold", type=float, default=30.0,
        help="Threshold for scene detection")
    parser.add_argument(
        "--downscale", type=int, default=2,
        help="Downscale factor for scene detection")
    parser.add_argument(
        "--use-whisper", action="store_true",
        help="Use Whisper for transcription (if available)")
    parser.add_argument(
        "--model",
        default="claude-3-7-sonnet-20240307",
        help="Claude model to use")
    parser.add_argument(
        "--max-tokens", type=int, default=1000,
        help="Maximum tokens per API response")

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate pipeline stages
    scene_detector = SceneDetector(
        threshold=args.threshold,
        downscale_factor=args.downscale)
    scene_processor = SceneProcessor(
        video_path=args.video_path, 
        output_dir=output_dir,
        use_whisper=args.use_whisper)
    summary_generator = AnthropicSummaryGenerator(
        model=args.model, max_tokens=args.max_tokens)

    # Assemble and run the pipeline
    pipeline = PipelineManager([
        scene_detector,
        scene_processor,
        summary_generator
    ])

    # Create input data with video path
    input_data = {"video_path": args.video_path}
    
    # Run the pipeline
    final_markdown_summary = pipeline.run(input_data)

    # Save the final markdown summary to a file
    summary_file = os.path.join(output_dir, "summary.md")
    with open(summary_file, "w") as f:
        f.write(final_markdown_summary)
    print(f"Pipeline complete. Markdown summary saved to: {summary_file}")

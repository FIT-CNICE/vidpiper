from scenedetect.video_manager import VideoManager
from scenedetect import detect, ContentDetector, SceneManager
import os
import subprocess
import json
import requests
import base64
import tempfile
import threading
import time
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

    If no scenes are detected within the timeout period (default 3 minutes),
    or if detected scenes are too large (>5MB), the video will be divided
    into appropriate segments automatically.
    """

    def __init__(self, threshold: float = 30.0, downscale_factor: int = 64,
                 min_scene_len: int = 15, timeout_seconds: int = 180,
                 max_size_mb: float = 5.0):
        self.threshold = threshold
        self.downscale_factor = downscale_factor
        self.min_scene_len = min_scene_len  # Minimum scene length in frames
        # Timeout for scene detection (default 3 min)
        self.timeout_seconds = timeout_seconds
        # Maximum scene size in MB (default 5MB)
        self.max_size_mb = max_size_mb

    def run(self, input_data: Dict) -> List[Scene]:
        video_path = input_data.get("video_path")
        if not video_path:
            raise ValueError("Input must contain 'video_path' key")

        print(
            f"Stage 1: Detecting scenes using PySceneDetect for {video_path}...")
        print(f"Checking if file exists: {os.path.exists(video_path)}")

        # Get video duration using ffprobe
        video_duration = self._get_video_duration(video_path)
        print(f"Video duration: {video_duration:.2f} seconds")

        # Attempt scene detection with timeout
        scenes = []
        detection_complete = False
        detection_thread = None

        def detect_scenes_thread():
            nonlocal scenes, detection_complete
            try:
                scenes = self._detect_scenes(video_path)
                detection_complete = True
            except Exception as e:
                print(f"Scene detection error: {e}")
                detection_complete = True

        # Start detection in a separate thread
        detection_thread = threading.Thread(target=detect_scenes_thread)
        detection_thread.daemon = True
        detection_thread.start()

        # Wait for detection to complete or timeout
        start_time = time.time()
        while not detection_complete and (
                time.time() - start_time) < self.timeout_seconds:
            time.sleep(1)

        if not detection_complete:
            print(
                f"Scene detection timed out after {self.timeout_seconds} seconds")
            scenes = []  # Will trigger fallback to manual segmentation
            # Don't stop the thread, just continue (it will be a daemon thread)

        # If no scenes detected or timeout occurred, divide video into equal
        # segments
        if not scenes:
            print(
                "No scenes detected or detection timed out. Dividing video into equal segments...")
            scenes = self._divide_video_into_segments(
                video_path, video_duration)

        # Check if any scene is too large and subdivide if needed
        scenes = self._ensure_max_scene_size(video_path, scenes)

        print(f"Final scene count: {len(scenes)}")
        return scenes

    def _detect_scenes(self, video_path: str) -> List[Scene]:
        """Core scene detection logic using PySceneDetect."""
        # Create a video manager and scene manager
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()

        # Add ContentDetector
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        # Set downscale factor to reduce memory usage
        video_manager.set_downscale_factor(self.downscale_factor)

        # Perform scene detection
        try:
            # Start video manager
            video_manager.start()

            # Detect scenes
            print("Detecting scenes...")
            scene_manager.detect_scenes(frame_source=video_manager)

            # Get scene list
            scene_list = scene_manager.get_scene_list()

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
        finally:
            video_manager.release()

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        return float(result.stdout.strip())

    def _get_scene_size_mb(self, video_path: str,
                           start_time: float, end_time: float) -> float:
        """Estimate scene size in MB based on duration and overall bitrate."""
        # Get video bitrate using ffprobe
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)

        # If bitrate is not available, use a conservative estimate (10 Mbps)
        try:
            bitrate = float(result.stdout.strip())
        except (ValueError, IndexError):
            bitrate = 10_000_000  # 10 Mbps as fallback

        # Calculate size: bitrate (bits/s) * duration (s) / 8 (bits/byte) /
        # 1024^2 (bytes/MB)
        duration = end_time - start_time
        size_mb = (bitrate * duration) / 8 / (1024 * 1024)
        return size_mb

    def _divide_video_into_segments(
            self, video_path: str, video_duration: float) -> List[Scene]:
        """Divide the video into equal segments under max_size_mb."""
        # Calculate how many segments we need
        # Estimate total video size and divide by max size per segment
        total_size_mb = self._get_scene_size_mb(video_path, 0, video_duration)
        num_segments = max(1, int(total_size_mb / self.max_size_mb) + 1)

        # Create equal segments
        segment_duration = video_duration / num_segments
        scenes = []

        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, video_duration)
            scenes.append(Scene(
                scene_id=i + 1,
                start=start_time,
                end=end_time
            ))

        print(f"Divided video into {len(scenes)} equal segments")
        return scenes

    def _ensure_max_scene_size(self, video_path: str,
                               scenes: List[Scene]) -> List[Scene]:
        """Ensure no scene exceeds the maximum size limit."""
        result_scenes = []
        next_scene_id = len(scenes) + 1

        for scene in scenes:
            # Calculate current scene size
            scene_size_mb = self._get_scene_size_mb(
                video_path, scene.start, scene.end)

            if scene_size_mb <= self.max_size_mb:
                # Scene is already small enough
                result_scenes.append(scene)
            else:
                # Scene needs to be subdivided
                duration = scene.end - scene.start
                # Calculate how many subdivisions needed
                num_parts = max(2, int(scene_size_mb / self.max_size_mb) + 1)
                part_duration = duration / num_parts

                print(
                    f"Subdividing scene {scene.scene_id} ({scene_size_mb:.2f}MB) into {num_parts} parts")

                for j in range(num_parts):
                    sub_start = scene.start + (j * part_duration)
                    sub_end = min(
                        scene.start + ((j + 1) * part_duration),
                        scene.end)

                    # For clarity, use original scene ID with part number
                    scene_id = next_scene_id
                    next_scene_id += 1

                    result_scenes.append(Scene(
                        scene_id=scene_id,
                        start=sub_start,
                        end=sub_end
                    ))

        # Re-number scene IDs sequentially for consistency
        for i, scene in enumerate(result_scenes):
            scene.scene_id = i + 1

        return result_scenes

# ---------------------------------------------------------
# Stage 2: Screenshot and Transcript Extraction
# ---------------------------------------------------------


class SceneProcessor(PipelineStage):
    """
    For each detected scene, extracts a screenshot (using FFmpeg)
    at the midpoint and generates a transcript.
    """

    def __init__(self, video_path: str, output_dir: str,
                 use_whisper: bool = False):
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

            print(
                f"Processed scene {scene.scene_id}: {scene.start:.2f}s to {scene.end:.2f}s")

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
            beam_size=1,  # Reduce beam size
            best_of=1    # Only return the top result
        )

        return result["text"].strip()

    def _placeholder_transcript(self, start: float, end: float) -> str:
        """Generate a placeholder transcript."""
        transcript = f"Transcript for scene from {start:.2f} to {end:.2f} seconds."
        print(
            f"Generated placeholder transcript for scene ({start:.2f}-{end:.2f}s).")
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
                print(
                    f"Running stage {i+1}/{len(self.stages)}: {stage.__class__.__name__}")
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
                print(
                    f"Error during cleanup of {stage.__class__.__name__}: {str(e)}")


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
        "--downscale", type=int, default=64,
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
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout in seconds for scene detection (default: 180)")
    parser.add_argument(
        "--max-size", type=float, default=5.0,
        help="Maximum size of scenes in MB (default: 5.0)")

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate pipeline stages
    scene_detector = SceneDetector(
        threshold=args.threshold,
        downscale_factor=args.downscale,
        timeout_seconds=args.timeout,
        max_size_mb=args.max_size)
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

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

    Supports skipping portions at the beginning and end of videos that
    might contain irrelevant content (intros, credits, etc.).
    """

    def __init__(self, threshold: float = 30.0, downscale_factor: int = 64,
                 min_scene_len: int = 15, timeout_seconds: int = 180,
                 max_size_mb: float = 20.0, skip_start: float = 0.0,
                 skip_end: float = 0.0, max_scene: int = None):
        self.threshold = threshold
        # Store initial threshold for adaptive adjustment
        self.initial_threshold = threshold
        self.downscale_factor = downscale_factor
        self.min_scene_len = min_scene_len  # Minimum scene length in frames
        # Timeout for scene detection (default 3 min)
        self.timeout_seconds = timeout_seconds
        # Maximum scene size in MB (default 20MB)
        self.max_size_mb = max_size_mb
        # Number of seconds to skip at the beginning of the video
        self.skip_start = skip_start
        # Number of seconds to skip at the end of the video
        self.skip_end = skip_end
        # Maximum number of scenes to detect
        self.max_scene = max_scene

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

        # Set max_scene based on video duration if not provided
        if self.max_scene is None:
            self.max_scene = max(1, int(video_duration / 120))
            print(
                f"Setting max_scene to {self.max_scene} based on video duration (assuming 120s per scene)")

        # Apply skip parameters
        effective_start = self.skip_start
        effective_end = video_duration - self.skip_end

        # Validate skip parameters
        if effective_start >= effective_end:
            raise ValueError(
                f"Invalid skip parameters: start ({self.skip_start}s) and end ({self.skip_end}s) " +
                f"would leave no content in the {video_duration:.2f}s video")

        print(
            f"Processing video from {effective_start:.2f}s to {effective_end:.2f}s " +
            f"(skipping first {self.skip_start:.2f}s and last {self.skip_end:.2f}s)")

        # Use adaptive threshold to control number of scenes
        max_attempts = 3  # Limit number of retry attempts
        current_attempt = 0
        scenes = []

        while current_attempt < max_attempts:
            # Attempt scene detection with timeout
            scenes = []
            detection_complete = False

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
                break  # Exit the retry loop

            scene_count = len(scenes)
            print(
                f"Detected {scene_count} scenes with threshold {self.threshold:.2f}")

            # Check if number of scenes is acceptable
            if scene_count <= self.max_scene or scene_count == 0:
                # Either we have fewer scenes than max_scene, or no scenes
                # detected
                break

            # Too many scenes detected, increase threshold for next attempt
            current_attempt += 1
            if current_attempt < max_attempts:
                # Calculate new threshold based on ratio of detected scenes to
                # max_scene
                ratio = scene_count / self.max_scene
                # Increase threshold proportionally to the ratio
                self.initial_threshold = self.threshold
                self.threshold = self.initial_threshold * \
                    (1 + (ratio - 1) * 0.8)
                print(
                    f"Too many scenes ({scene_count} > {self.max_scene}). "
                    f"Adjusting threshold to {self.threshold:.2f} "
                    f"(attempt {current_attempt}/{max_attempts})")

        # If no scenes detected after all attempts, divide video into equal
        # segments
        if not scenes:
            print(
                "No scenes detected or detection timed out. " +
                "Dividing video into equal segments...")
            scenes = self._divide_video_into_segments(
                video_path, video_duration)

        # If we still have too many scenes, select max_scene scenes by duration
        if len(scenes) > self.max_scene:
            print(
                f"Still detected {len(scenes)} scenes after "
                "threshold adjustment. Selecting all scenes.")
            scenes.sort(key=lambda x: x.end - x.start, reverse=True)
            # Re-number scene IDs sequentially for consistency
            for i, scene in enumerate(scenes):
                scene.scene_id = i + 1

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

        # Get video duration
        video_duration = self._get_video_duration(video_path)
        effective_start = self.skip_start
        effective_end = video_duration - self.skip_end

        # Perform scene detection
        try:
            # Start video manager
            video_manager.start()

            # Detect scenes
            print(
                f"Detecting scenes from {effective_start:.2f}s to {effective_end:.2f}s...")
            scene_manager.detect_scenes(frame_source=video_manager)

            # Get scene list
            scene_list = scene_manager.get_scene_list()

            # Convert to our Scene dataclass format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()

                # Only include scenes that fall within our effective range
                if end_time > effective_start and start_time < effective_end:
                    # Clip scenes to effective range if needed
                    start_time = max(start_time, effective_start)
                    end_time = min(end_time, effective_end)

                    scenes.append(Scene(
                        scene_id=i + 1,
                        start=start_time,
                        end=end_time
                    ))

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
        """Divide the video into equal segments under max_size_mb, respecting skip parameters."""
        # Apply skip parameters
        effective_start = self.skip_start
        effective_end = video_duration - self.skip_end
        effective_duration = effective_end - effective_start

        if effective_duration <= 0:
            raise ValueError(
                f"Invalid skip parameters: start ({self.skip_start}s) and end ({self.skip_end}s) " +
                f"would leave no content in the {video_duration:.2f}s video")

        # Calculate how many segments we need based on effective duration
        total_size_mb = self._get_scene_size_mb(
            video_path, effective_start, effective_end)
        num_segments = max(1, int(total_size_mb / self.max_size_mb) + 1)

        # Create equal segments within the effective range
        segment_duration = effective_duration / num_segments
        scenes = []

        for i in range(num_segments):
            start_time = effective_start + (i * segment_duration)
            end_time = min(
                effective_start + ((i + 1) * segment_duration),
                effective_end)
            scenes.append(Scene(
                scene_id=i + 1,
                start=start_time,
                end=end_time
            ))

        print(
            f"Divided video into {len(scenes)} equal segments within the specified time range")
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
    at the midpoint and generates a transcript using Whisper when available.
    """

    def __init__(self, video_path: str, output_dir: str,
                 use_whisper: bool = True, whisper_model: str = "small"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.use_whisper = use_whisper
        self.whisper_model = whisper_model
        self.temp_files = []
        self.whisper_model_instance = None
        self.detected_language = None

        # Verify video path exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

        # Check if whisper is available and load model at initialization
        if self.use_whisper:
            self._check_whisper_availability_and_load_model()

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

    def _check_whisper_availability_and_load_model(self):
        """
        Check if Whisper is available, select the appropriate model,
        and load it."""
        try:
            import whisper
            import torch

            # Check available VRAM to determine the appropriate model size
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(
                    0).total_memory / 1e9
                print(f"Found GPU with {vram_gb:.2f} GB VRAM")

                # Select model based on available VRAM
                if vram_gb > 5:
                    self.whisper_model = "medium"
                    print("Using medium Whisper model")
                elif vram_gb > 2:
                    self.whisper_model = "small"
                    print("Using small Whisper model")
                elif vram_gb > 1:
                    self.whisper_model = "base"
                    print("Using base Whisper model")
                else:
                    self.whisper_model = "tiny"
                    print("Using tiny Whisper model due to limited VRAM")
            else:
                print("No GPU detected, using tiny Whisper model on CPU")
                self.whisper_model = "tiny"

            # Load the model once to use for language detection
            print(
                f"Loading initial {self.whisper_model} Whisper model for language detection...")
            self.whisper_model_instance = whisper.load_model(
                self.whisper_model, device="cuda"
                if torch.cuda.is_available() else "cpu")

            # Detect language from the middle of the video
            self._detect_language()

            # If English is detected, reload with .en model for better accuracy
            if self.detected_language == "en":
                model_name = f"{self.whisper_model}.en"
                print(f"English detected, switching to {model_name} model")

                # Reload the model with .en suffix
                self.whisper_model_instance = whisper.load_model(
                    model_name, device="cuda"
                    if torch.cuda.is_available() else "cpu")
                print(f"Successfully loaded {model_name} Whisper model")
            else:
                print(
                    f"Using {self.whisper_model} Whisper model for detected language: {self.detected_language or 'unknown'}")

        except ImportError:
            print("Whisper not installed. Will use placeholder transcripts.")
            self.use_whisper = False
        except Exception as e:
            print(f"Error checking Whisper availability or loading model: {e}")
            self.use_whisper = False

    def _detect_language(self):
        """Detect language from a sample audio clip at the middle of the video."""
        try:
            import whisper

            # Get video duration using ffprobe
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.video_path
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
            video_duration = float(result.stdout.strip())

            # Extract a small audio clip from the middle of the video for
            # language detection
            middle_time = video_duration / 2
            sample_audio_path = self._extract_audio_clip(
                max(0, middle_time - 15),
                # 30 second clip centered at the middle
                min(video_duration, middle_time + 15))

            print(
                f"Detecting language from middle of video (around {middle_time:.2f}s)...")
            audio = whisper.load_audio(sample_audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(
                self.whisper_model_instance.device)

            _, probs = self.whisper_model_instance.detect_language(mel)
            self.detected_language = max(probs, key=probs.get)
            print(
                f"Detected language: {self.detected_language} (confidence: {probs[self.detected_language]:.2f})")

        except Exception as e:
            print(f"Error detecting language: {e}")
            self.detected_language = None  # Default to auto-detection if this fails

    def extract_screenshot(self, timestamp: float, output_path: str) -> None:
        """Extract a screenshot from the video at the specified timestamp."""
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", self.video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
            "-y"  # Overwrite output file if exists
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            print(f"Extracted screenshot at {timestamp:.2f}s to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting screenshot at {timestamp:.2f}s: {e}")
            print(f"STDERR: {e.stderr}")
            # Continue processing despite error

    def extract_transcript(self, start: float, end: float) -> str:
        """Extract transcript for the specified time range."""
        if self.use_whisper and self.whisper_model_instance is not None:
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

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            return output_path
        except subprocess.CalledProcessError as e:
            print(
                f"Error extracting audio for scene ({start:.2f}-{end:.2f}s): {e}")
            raise

    def _transcribe_with_whisper(self, start: float, end: float) -> str:
        """Transcribe audio using Whisper with the pre-loaded model and detected language."""
        # Extract audio clip for this scene
        try:
            audio_clip_path = self._extract_audio_clip(start, end)
        except subprocess.CalledProcessError:
            return self._placeholder_transcript(start, end)

        try:
            import whisper
            import torch

            # Run transcription with the detected language
            transcription_options = {
                "fp16": torch.cuda.is_available(),  # Use half-precision on GPU
                "beam_size": 5,  # Increase beam size for better accuracy
                "best_of": 5,  # Consider more candidates for better results
                "task": "transcribe"  # Force transcription task
            }

            # Add language parameter only if we have detected one
            if self.detected_language:
                transcription_options["language"] = self.detected_language

            result = self.whisper_model_instance.transcribe(
                audio_clip_path, **transcription_options
            )

            transcript = result["text"].strip()
            lang_info = f"in {self.detected_language}" if self.detected_language else ""
            print(
                f"Successfully transcribed audio {lang_info} ({start:.2f}-{end:.2f}s)")
            return transcript

        except Exception as e:
            print(f"Error during transcription with Whisper: {e}")
            return self._placeholder_transcript(start, end)

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

    This implementation uses the Anthropic Python client with multimodal support
    to process both screenshots and transcripts for each scene, with optimizations
    for token usage and focus on visual content.
    """

    def __init__(self, model: str = "claude-3-7-sonnet-20250219",
                 max_tokens: int = 2000, output_dir: str = "output"):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required. Install it with 'pip install anthropic'.")

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.output_dir = output_dir  # Store the output directory
        self.summaries = {}  # Store individual scene summaries
        self.previous_summary_context = ""  # Track previous summary for continuity
        self.processed_scenes_count = 0  # Track number of processed scenes

        # System prompt for technical content summarization balanced for
        # general audience
        self.system_prompt = (
            "You are a technical content summarizer specializing in creating accessible, "
            "accurate summaries of technical presentations and demos for general audiences. "
            "Use BOTH THE VISUAL CONTENT AND TRANSCRIPT to create comprehensive summaries. "
            "Transcript CONTAINS typo of technical jargons"
            "For any technical terms or concepts, provide intuitive explanations that are "
            "accessible to non-experts while maintaining technical accuracy. "
            "Identify key technical concepts, data points, and main arguments from both the visuals "
            "and the transcript, explaining complex ideas using analogies or simplified examples when appropriate. "
            "Create logically connected summaries that flow naturally from previous scenes while ensuring "
            "technical content is understandable to a broad audience.")

    def run(self, processed_scenes: List[Scene]) -> str:
        print("Stage 3: Generating summaries with Claude 3.7...")
        complete_summary = "# Video Summary\n\n"

        for i, scene in enumerate(processed_scenes):
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

            # Add screenshot reference with a relative path
            screenshot_filename = os.path.basename(screenshot_path)
            # Use relative path for screenshots in the markdown
            relative_screenshot_path = f"./screenshots/{screenshot_filename}"
            screenshot_ref = f"![Scene {scene_id} Screenshot]({relative_screenshot_path})\n\n"

            # Get summary for this scene using multimodal API with awareness of
            # previous content
            scene_summary = self._generate_scene_summary(
                scene_id, screenshot_path, transcript, start_time, end_time,
                i, len(processed_scenes))

            # Add this scene to the complete summary
            complete_summary += scene_heading
            complete_summary += screenshot_ref
            complete_summary += scene_summary

            # Only add separator if not the last scene
            if i < len(processed_scenes) - 1:
                complete_summary += "\n\n---\n\n"

            # Save progress incrementally
            self.summaries[scene_id] = scene_summary
            self.processed_scenes_count += 1

            # Update previous context for continuity (limit to last 2 summaries
            # to control token usage)
            self.previous_summary_context = scene_summary

            # Save the current state of the summary in the output directory
            in_progress_file = os.path.join(
                self.output_dir, "summary_in_progress.md")
            with open(in_progress_file, "w", encoding="utf-8") as f:
                f.write(complete_summary)

        return complete_summary

    def _generate_overall_introduction(
            self, initial_scenes: List[Scene]) -> str:
        """Generate an overall introduction based on the first few scenes."""
        # Method kept for compatibility but no longer used
        return ""

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 for API request."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _generate_scene_summary(
            self, scene_id: int, screenshot_path: str, transcript: str,
            start_time: float, end_time: float, scene_index: int,
            total_scenes: int) -> str:
        """
        Generate a summary for a single scene using Claude 3.7's
        multimodal capabilities with retry mechanism for handling API errors.
        """
        # Encode the screenshot as base64
        base64_image = self._encode_image(screenshot_path)

        # Create the user prompt with different strategies depending on
        # position
        if scene_index == 0:
            # First scene - establish the topic
            context_directive = (
                "As this is the first scene, " +
                "establish the main topic and context of the presentation." +
                "if the image content does not look like part of a demo, " +
                "presentation, or panel discussion. Tell me using EXACTLY " +
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

        # Retry parameters
        max_retries = 3
        retry_delay = 2  # Initial delay in seconds
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Create message with the Anthropic client
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=min(self.max_tokens, 3000),
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": user_prompt
                                }
                            ]
                        }
                    ]
                )

                # Extract the response text
                generated_text = message.content[0].text

                generated_text = generated_text.replace("\n## ", "\n### ")
                generated_text = generated_text.replace("\n# ", "\n## ")

                print(f"Generated summary for scene {scene_id}")
                return generated_text

            except Exception as e:
                retry_count += 1
                error_msg = "Error generating summary for scene " + \
                    f"{scene_id} (attempt {retry_count}/{max_retries}):" + \
                    f"{str(e)}"
                print(error_msg)

                if retry_count < max_retries:
                    # Exponential backoff with jitter
                    import random
                    # Add random jitter between 0-0.5 seconds
                    jitter = random.uniform(0, 0.5)
                    sleep_time = (
                        retry_delay * (2 ** (retry_count - 1))) + jitter
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    # All retries failed, return error message
                    return "*Error generating summary for scene " + \
                        f"{scene_id} after {max_retries} attempts: {str(e)}*"


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
        "--threshold", type=float, default=35.0,
        help="Threshold for scene detection")
    parser.add_argument(
        "--downscale", type=int, default=64,
        help="Downscale factor for scene detection")
    parser.add_argument(
        "--use-whisper", action="store_true",
        help="Use Whisper for transcription (if available)")
    parser.add_argument(
        "--model",
        default="claude-3-7-sonnet-20250219",
        help="Claude model to use")
    parser.add_argument(
        "--max-tokens", type=int, default=1500,
        help="Maximum tokens per API response")
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout in seconds for scene detection (default: 180)")
    parser.add_argument(
        "--max-size", type=float, default=20.0,
        help="Maximum size of scenes in MB (default: 20.0)")
    parser.add_argument(
        "--skip-start", type=float, default=60.0,
        help="Number of seconds to skip at the beginning of the video (default: 60.0)")
    parser.add_argument(
        "--skip-end", type=float, default=0.0,
        help="Number of seconds to skip at the end of the video (default: 0.0)")
    parser.add_argument(
        "--max-scene", type=int, default=None,
        help="Maximum number of scenes to detect. If None, uses video length / 120 seconds per scene.")

    args = parser.parse_args()

    # Create output directory based on video filename if not specified
    if args.output_dir == "output":
        # Extract the base filename without extension
        video_basename = os.path.basename(args.video_path)
        video_name = os.path.splitext(video_basename)[0]
        output_dir = f"{video_name}_output"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Instantiate pipeline stages
    scene_detector = SceneDetector(
        threshold=args.threshold,
        downscale_factor=args.downscale,
        timeout_seconds=args.timeout,
        max_size_mb=args.max_size,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        max_scene=args.max_scene)
    scene_processor = SceneProcessor(
        video_path=args.video_path,
        output_dir=output_dir,
        use_whisper=True)  # Now using Whisper by default
    summary_generator = AnthropicSummaryGenerator(
        model=args.model,
        max_tokens=args.max_tokens,
        output_dir=output_dir)

    # Assemble and run the pipeline
    pipeline = PipelineManager([
        scene_detector,
        scene_processor,
        summary_generator
    ])

    # Create input data with video path
    input_data = {"video_path": args.video_path}

    try:
        # Run the pipeline
        final_markdown_summary = pipeline.run(input_data)

        # Save the final markdown summary to a file with video name + _sum
        # suffix
        video_basename = os.path.basename(args.video_path)
        video_name = os.path.splitext(video_basename)[0]
        summary_file = os.path.join(output_dir, f"{video_name}_sum.md")

        with open(summary_file, "w") as f:
            f.write(final_markdown_summary)
        print(f"Pipeline complete. Markdown summary saved to: {summary_file}")

        # Clean up the in-progress summary file if it exists
        in_progress_file = os.path.join(output_dir, "summary_in_progress.md")
        if os.path.exists(in_progress_file):
            os.remove(in_progress_file)
            print(f"Removed temporary file: {in_progress_file}")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise

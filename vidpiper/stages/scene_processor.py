"""Scene processing stage for extracting screenshots and transcripts."""

import os
import subprocess
import tempfile

from ..core.pipeline import PipelineStage
from ..core.data_classes import PipelineResult


class SceneProcessor(PipelineStage):
    """
    For each detected scene, extracts a screenshot (using FFmpeg)
    at the midpoint and generates a transcript using Whisper when available.
    """

    def __init__(
        self,
        output_dir: str,
        use_whisper: bool = True,
        whisper_model: str = "small",
    ):
        self.output_dir = output_dir
        self.use_whisper = use_whisper
        self.whisper_model = whisper_model
        self.temp_files = []
        self.whisper_model_instance = None
        self.detected_language = None

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

    def run(self, data: PipelineResult) -> PipelineResult:
        video_path = data.video_path
        scenes = data.scenes

        if not video_path:
            raise ValueError("Input must contain video_path")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not scenes:
            raise ValueError("No scenes to process")

        # Update output directory in the result
        data.output_dir = self.output_dir

        # Check if whisper is available and load model
        if self.use_whisper:
            self._check_whisper_availability_and_load_model(video_path)

        print(
            f"Extracting screenshots and transcripts for {len(scenes)} scenes"
        )

        for scene in scenes:
            # Generate screenshot filename
            screenshot_path = os.path.join(
                self.output_dir, "screenshots", f"scene_{scene.scene_id}.jpg"
            )

            # Choose a timestamp at the midpoint of the scene
            timestamp = (scene.start + scene.end) / 2

            # Extract screenshot and transcript
            self.extract_screenshot(video_path, timestamp, screenshot_path)
            transcript = self.extract_transcript(
                video_path, scene.start, scene.end
            )

            # Update scene object
            scene.screenshot = screenshot_path
            scene.transcript = transcript

            print(
                f"Processed scene {scene.scene_id}: {scene.start:.2f}s to {scene.end:.2f}s"
            )

        # Update scenes in the pipeline result
        data.scenes = scenes

        # Add metadata about processing
        data.metadata["scene_processing"] = {
            "whisper_available": self.use_whisper
            and self.whisper_model_instance is not None,
            "detected_language": self.detected_language,
            "whisper_model": self.whisper_model if self.use_whisper else None,
        }

        return data

    def _check_whisper_availability_and_load_model(self, video_path: str):
        """
        Check if Whisper is available, select the appropriate model,
        and load it."""
        try:
            import whisper
            import torch

            # Check available VRAM to determine the appropriate model size
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"Found GPU with {vram_gb:.2f} GB VRAM")

                # Select model based on available VRAM
                if vram_gb > 10:
                    self.whisper_model = "large"
                    print("Using large Whisper model")
                elif vram_gb > 6:
                    self.whisper_model = "turbo"
                    print("Using turbo Whisper model")
                elif vram_gb > 5:
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
                f"Loading initial {self.whisper_model} Whisper model for language detection..."
            )
            self.whisper_model_instance = whisper.load_model(
                self.whisper_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Detect language from the middle of the video
            self._detect_language(video_path)

            # If English is detected, reload with .en model for better accuracy
            if self.detected_language == "en":
                model_name = f"{self.whisper_model}.en"
                print(f"English detected, switching to {model_name} model")

                # Reload the model with .en suffix
                self.whisper_model_instance = whisper.load_model(
                    model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                print(f"Successfully loaded {model_name} Whisper model")
            else:
                print(
                    f"Using {self.whisper_model} Whisper model for detected language: {self.detected_language or 'unknown'}"
                )

        except ImportError:
            print("Whisper not installed. Will use placeholder transcripts.")
            self.use_whisper = False
        except Exception as e:
            print(f"Error checking Whisper availability or loading model: {e}")
            self.use_whisper = False

    def _detect_language(self, video_path: str):
        """Detect language from a sample audio clip at the middle of the video."""
        try:
            import whisper

            # Get video duration using ffprobe
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            video_duration = float(result.stdout.strip())

            # Extract a small audio clip from the middle of the video for
            # language detection
            middle_time = video_duration / 2
            sample_audio_path = self._extract_audio_clip(
                video_path,
                max(0, middle_time - 300),
                # 600 second clip centered at the middle
                min(video_duration, middle_time + 300),
            )

            print(
                f"Detecting language from middle of video (around {middle_time:.2f}s)..."
            )
            audio = whisper.load_audio(sample_audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(
                self.whisper_model_instance.device
            )

            _, probs = self.whisper_model_instance.detect_language(mel)
            self.detected_language = max(probs, key=probs.get)
            print(
                f"Detected language: {self.detected_language} (confidence: {probs[self.detected_language]:.2f})"
            )

        except Exception as e:
            print(f"Error detecting language: {e}")
            self.detected_language = (
                None  # Default to auto-detection if this fails
            )

    def extract_screenshot(
        self, video_path: str, timestamp: float, output_path: str
    ) -> None:
        """Extract a screenshot from the video at the specified timestamp."""
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-q:v",
            "2",
            output_path,
            "-y",  # Overwrite output file if exists
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            print(f"Extracted screenshot at {timestamp:.2f}s to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting screenshot at {timestamp:.2f}s: {e}")
            print(f"STDERR: {e.stderr}")
            # Continue processing despite error

    def extract_transcript(
        self, video_path: str, start: float, end: float
    ) -> str:
        """Extract transcript for the specified time range."""
        if self.use_whisper and self.whisper_model_instance is not None:
            try:
                return self._transcribe_with_whisper(video_path, start, end)
            except Exception as e:
                print(f"Whisper transcription failed: {e}")
                return self._placeholder_transcript(start, end)
        else:
            return self._placeholder_transcript(start, end)

    def _extract_audio_clip(
        self, video_path: str, start: float, end: float
    ) -> str:
        """Extract audio clip for a scene using ffmpeg."""
        output_path = tempfile.mktemp(suffix=".wav")
        self.temp_files.append(output_path)

        duration = end - start
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path,
        ]

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return output_path
        except subprocess.CalledProcessError as e:
            print(
                f"Error extracting audio for scene ({start:.2f}-{end:.2f}s): {e}"
            )
            raise

    def _transcribe_with_whisper(
        self, video_path: str, start: float, end: float
    ) -> str:
        """Transcribe audio using Whisper with the pre-loaded model and detected language."""
        # Extract audio clip for this scene
        try:
            audio_clip_path = self._extract_audio_clip(video_path, start, end)
        except subprocess.CalledProcessError:
            return self._placeholder_transcript(start, end)

        try:
            import torch

            # Run transcription with the detected language
            transcription_options = {
                "fp16": torch.cuda.is_available(),  # Use half-precision on GPU
                "beam_size": 5,  # Increase beam size for better accuracy
                "best_of": 5,  # Consider more candidates for better results
                "task": "transcribe",  # Force transcription task
            }

            # Add language parameter only if we have detected one
            if self.detected_language:
                transcription_options["language"] = self.detected_language

            result = self.whisper_model_instance.transcribe(
                audio_clip_path, **transcription_options
            )

            transcript = result["text"].strip()
            lang_info = (
                f"in {self.detected_language}" if self.detected_language else ""
            )
            print(
                f"Successfully transcribed audio {lang_info} ({start:.2f}-{end:.2f}s)"
            )
            return transcript

        except Exception as e:
            print(f"Error during transcription with Whisper: {e}")
            return self._placeholder_transcript(start, end)

    def _placeholder_transcript(self, start: float, end: float) -> str:
        """Generate a placeholder transcript."""
        transcript = (
            f"Transcript for scene from {start:.2f} to {end:.2f} seconds."
        )
        print(
            f"Generated placeholder transcript for scene ({start:.2f}-{end:.2f}s)."
        )
        return transcript

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove temp file {file_path}: {e}")


def create_scene_processor(
    output_dir: str, use_whisper: bool = True, whisper_model: str = "small"
) -> SceneProcessor:
    """Factory function to create a scene processor stage."""
    return SceneProcessor(
        output_dir=output_dir,
        use_whisper=use_whisper,
        whisper_model=whisper_model,
    )

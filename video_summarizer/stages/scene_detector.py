"""Scene detection stage for the video summarizer pipeline."""

import os
import subprocess
import threading
import time
from typing import List
from scenedetect.video_manager import VideoManager
from scenedetect import ContentDetector, SceneManager

from ..core.pipeline import PipelineStage
from ..core.data_classes import Scene, PipelineResult


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

    def __init__(
        self,
        threshold: float = 30.0,
        downscale_factor: int = 64,
        min_scene_len: int = 15,
        timeout_seconds: int = 180,
        max_size_mb: float = 20.0,
        skip_start: float = 0.0,
        skip_end: float = 0.0,
        max_scene: int = None,
    ):
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

    def run(self, data: PipelineResult) -> PipelineResult:
        video_path = data.video_path
        if not video_path:
            raise ValueError("Input must contain video_path")

        print(f"Detecting scenes using PySceneDetect for {video_path}...")
        print(f"Checking if file exists: {os.path.exists(video_path)}")

        # Get video duration using ffprobe
        video_duration = self._get_video_duration(video_path)
        print(f"Video duration: {video_duration:.2f} seconds")

        # Set max_scene based on video duration if not provided
        if self.max_scene is None:
            self.max_scene = max(1, int(video_duration / 100))
            print(
                f"Setting max_scene to {self.max_scene} based on "
                "video duration(assuming 100s per scene)"
            )

        # Apply skip parameters
        effective_start = self.skip_start
        effective_end = video_duration - self.skip_end

        # Validate skip parameters
        if effective_start >= effective_end:
            raise ValueError(
                f"Invalid skip parameters: start ({self.skip_start}s) "
                f"and end ({self.skip_end}s) "
                f"would leave no content in the {video_duration:.2f}s video"
            )

        print(
            f"Processing video from {effective_start:.2f}s "
            f"to {effective_end:.2f}s "
            f"(skipping first {self.skip_start:.2f}s and "
            f"last {self.skip_end:.2f}s)"
        )

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
            while (
                not detection_complete
                and (time.time() - start_time) < self.timeout_seconds
            ):
                time.sleep(1)

            if not detection_complete:
                print(
                    f"Scene detection timed out after {self.timeout_seconds} seconds"
                )
                scenes = []  # Will trigger fallback to manual segmentation
                break  # Exit the retry loop

            scene_count = len(scenes)
            print(
                f"Detected {scene_count} scenes with threshold {self.threshold:.2f}"
            )

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
                self.threshold = self.initial_threshold * (
                    1 + (ratio - 1) * 0.8
                )
                print(
                    f"Too many scenes ({scene_count} > {self.max_scene}). "
                    f"Adjusting threshold to {self.threshold:.2f} "
                    f"(attempt {current_attempt}/{max_attempts})"
                )

        # If no scenes detected after all attempts, divide video into equal
        # segments
        if not scenes:
            print(
                "No scenes detected or detection timed out. "
                + "Dividing video into equal segments..."
            )
            scenes = self._divide_video_into_segments(
                video_path, video_duration
            )

        # If we still have too many scenes, select max_scene scenes by duration
        if len(scenes) > self.max_scene:
            print(
                f"Still detected {len(scenes)} scenes after "
                "threshold adjustment. Selecting all scenes."
            )
            scenes.sort(key=lambda x: x.end - x.start, reverse=True)
            # Re-number scene IDs sequentially for consistency
            for i, scene in enumerate(scenes):
                scene.scene_id = i + 1

        # Check if any scene is too large and subdivide if needed
        scenes = self._ensure_max_scene_size(video_path, scenes)

        print(f"Final scene count: {len(scenes)}")

        # Store scenes in the pipeline result
        data.scenes = scenes

        # Add some metadata about the detection process
        data.metadata["scene_detection"] = {
            "threshold": self.threshold,
            "original_scene_count": scene_count
            if "scene_count" in locals()
            else 0,
            "final_scene_count": len(scenes),
            "effective_duration": effective_end - effective_start,
        }

        return data

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
                f"Detecting scenes from {effective_start:.2f}s to {effective_end:.2f}s..."
            )
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

                    scenes.append(
                        Scene(scene_id=i + 1, start=start_time, end=end_time)
                    )

            return scenes
        finally:
            video_manager.release()

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
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
        return float(result.stdout.strip())

    def _get_scene_size_mb(
        self, video_path: str, start_time: float, end_time: float
    ) -> float:
        """Estimate scene size in MB based on duration and overall bitrate."""
        # Get video bitrate using ffprobe
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

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
        self, video_path: str, video_duration: float
    ) -> List[Scene]:
        """Divide the video into equal segments under max_size_mb, respecting skip parameters."""
        # Apply skip parameters
        effective_start = self.skip_start
        effective_end = video_duration - self.skip_end
        effective_duration = effective_end - effective_start

        if effective_duration <= 0:
            raise ValueError(
                f"Invalid skip parameters: start ({self.skip_start}s) and end ({self.skip_end}s) "
                + f"would leave no content in the {video_duration:.2f}s video"
            )

        # Calculate how many segments we need based on effective duration
        total_size_mb = self._get_scene_size_mb(
            video_path, effective_start, effective_end
        )
        num_segments = max(1, int(total_size_mb / self.max_size_mb) + 1)

        # Create equal segments within the effective range
        segment_duration = effective_duration / num_segments
        scenes = []

        for i in range(num_segments):
            start_time = effective_start + (i * segment_duration)
            end_time = min(
                effective_start + ((i + 1) * segment_duration), effective_end
            )
            scenes.append(Scene(scene_id=i + 1, start=start_time, end=end_time))

        print(
            f"Divided video into {len(scenes)} equal segments within the specified time range"
        )
        return scenes

    def _ensure_max_scene_size(
        self, video_path: str, scenes: List[Scene]
    ) -> List[Scene]:
        """Ensure no scene exceeds the maximum size limit."""
        result_scenes = []
        next_scene_id = len(scenes) + 1

        for scene in scenes:
            # Calculate current scene size
            scene_size_mb = self._get_scene_size_mb(
                video_path, scene.start, scene.end
            )

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
                    f"Subdividing scene {scene.scene_id} ({scene_size_mb:.2f}MB) into {num_parts} parts"
                )

                for j in range(num_parts):
                    sub_start = scene.start + (j * part_duration)
                    sub_end = min(
                        scene.start + ((j + 1) * part_duration), scene.end
                    )

                    # For clarity, use original scene ID with part number
                    scene_id = next_scene_id
                    next_scene_id += 1

                    result_scenes.append(
                        Scene(scene_id=scene_id, start=sub_start, end=sub_end)
                    )

        # Re-number scene IDs sequentially for consistency
        for i, scene in enumerate(result_scenes):
            scene.scene_id = i + 1

        return result_scenes


def create_scene_detector(
    threshold: float = 30.0,
    downscale_factor: int = 64,
    min_scene_len: int = 15,
    timeout_seconds: int = 180,
    max_size_mb: float = 20.0,
    skip_start: float = 0.0,
    skip_end: float = 0.0,
    max_scene: int = None,
) -> SceneDetector:
    """Factory function to create a scene detector stage."""
    return SceneDetector(
        threshold=threshold,
        downscale_factor=downscale_factor,
        min_scene_len=min_scene_len,
        timeout_seconds=timeout_seconds,
        max_size_mb=max_size_mb,
        skip_start=skip_start,
        skip_end=skip_end,
        max_scene=max_scene,
    )

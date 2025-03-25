# Video Summarizer

A modular pipeline for generating text summaries of video content with screenshots and transcripts.

## Features

- **Modular Pipeline Architecture**: Split the video summarization workflow into independent stages that can be run separately or as a complete pipeline.
- **Scene Detection**: Automatically detect scene changes in videos using PySceneDetect.
- **Scene Processing**: Extract screenshots and transcripts for each scene.
- **Summary Generation**: Use LLMs (Anthropic Claude, OpenAI GPT-4, Google Gemini) to generate summaries of each scene.
- **Extensible Design**: Create custom pipeline stages and insert them at any point in the pipeline.
- **Checkpoint System**: Save and load the state of the pipeline at any point to enable pausing and resuming processing.

## Requirements

- Python 3.8+
- FFmpeg installed on the system
- GPU recommended for faster processing (especially for Whisper transcription)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
# Only one of these is required
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Complete Pipeline

Run the complete pipeline with the CLI tool:

```bash
# Basic usage
python vid_summerizer_cli.py /path/to/video.mp4

# With options
python vid_summerizer_cli.py /path/to/video.mp4 \
  --output-dir my_output \
  --threshold 30.0 \
  --skip-start 60.0 \
  --use-whisper \
  --llm-provider anthropic
```

### Individual Stages

Run each stage separately:

1. Scene Detection:

```bash
# Detect scenes and save to JSON
python detect_scenes.py /path/to/video.mp4 \
  --threshold 35.0 \
  --skip-start 60.0
```

2. Scene Processing:

```bash
# Process scenes (extract screenshots and transcripts)
python process_scenes.py \
  --input-file video_output/detected_scenes.json \
  --use-whisper
```

3. Summary Generation:

```bash
# Generate summary
python generate_summary.py \
  --input-file video_output/processed_scenes.json \
  --llm-provider gemini
```

### Custom Pipelines

Create custom pipelines by extending the base classes:

```python
from video_summarizer.core import Pipeline, PipelineStage, PipelineResult
from video_summarizer.stages import create_scene_detector, create_scene_processor

# Create a custom stage
class MyCustomStage(PipelineStage):
    def run(self, data: PipelineResult) -> PipelineResult:
        # Custom processing logic here
        return data

# Create a pipeline with custom stages
pipeline = Pipeline()
pipeline.add_stage(create_scene_detector())
pipeline.add_stage(MyCustomStage())
pipeline.add_stage(create_scene_processor(output_dir="output"))

# Run the pipeline
result = pipeline.run(PipelineResult(video_path="/path/to/video.mp4"))
```

See `custom_pipeline_example.py` for a complete example of a custom pipeline.

## Module Structure

```
video_summarizer/
  ├── core/               # Core pipeline components
  │   ├── data_classes.py # Data classes for pipeline stages
  │   └── pipeline.py     # Pipeline infrastructure
  ├── stages/             # Pipeline stage implementations
  │   ├── scene_detector.py   # Scene detection stage
  │   ├── scene_processor.py  # Screenshot and transcript extraction
  │   └── summary_generator.py # Summary generation with LLMs
  └── llm_providers/      # LLM API provider implementations
      ├── base.py         # Base LLM provider interface
      ├── anthropic_provider.py # Claude implementation
      ├── openai_provider.py    # GPT-4 implementation
      └── gemini_provider.py    # Gemini implementation
```

## Customization Options

### Scene Detection

- **Threshold**: Sensitivity of scene detection (lower = more scenes)
- **Downscale**: Reduce resolution for faster processing
- **Skip Start/End**: Ignore portions at the beginning/end of the video
- **Max Size**: Maximum size in MB for each scene
- **Max Scene**: Maximum number of scenes to detect

### Scene Processing

- **Use Whisper**: Enable/disable transcript generation with Whisper
- **Whisper Model**: Choose model size (tiny, base, small, medium, large)

### Summary Generation

- **LLM Provider**: Choose which AI model to use (anthropic, openai, gemini)
- **Model**: Specific model name
- **Max Tokens**: Maximum tokens per API request

## Sample Output

The output includes:
- JSON files with scene metadata
- Screenshots for each scene
- A Markdown summary with screenshots and scene descriptions
- (Optional) Checkpoint files for resuming processing

## License

MIT
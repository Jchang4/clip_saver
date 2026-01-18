# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Clip Saver is a Python package for processing video streams (RTSP or files) with YOLO object detection and customizable callbacks. It uses Ultralytics YOLO for detection and Supervision for detection data structures.

## Installation & Development

```bash
# Install from source
pip install -e .

# Dependencies: lapx, supervision, ultralytics
```

## Architecture

### Core Components

**ClipSaver** (`clip_saver/clip_saver.py`): Main orchestrator that:
- Loads a YOLO model and runs inference on video frames
- Converts YOLO results to `supervision.Detections` format
- Runs callbacks in a separate process via multiprocessing Queue
- Supports stopping via `CLIP_SAVER_STOP` environment variable

**Video Sources** (`clip_saver/video_source/`): Abstract `BaseVideoSource` with implementations:
- `RTSPVideoSource`: Single RTSP stream
- `MultiRTSPVideoSource`: Multiple RTSP streams (writes to `list.streams` file)
- `FileVideoSource`: Local video files

**Callbacks** (`clip_saver/callbacks/`): Abstract `BaseCallback` with three lifecycle methods:
- `start()`: Initialize resources
- `run(frame: Frame)`: Process each frame
- `stop()`: Cleanup resources

**Frame** (`clip_saver/datatypes/frame.py`): Dataclass containing:
- `image`: numpy array (BGR format from OpenCV)
- `detections`: supervision.Detections object
- `timestamp`: ISO formatted string
- `video_path`: source path

### Detection Filtering

Filters are functions with signature `(sv.Detections, list[str]) -> sv.Detections` where the second argument is classnames. Multiple filters are applied sequentially.

### Multiprocessing Model

ClipSaver runs callbacks in a separate process. The main process handles YOLO inference and puts Frame objects into a Queue. The callback process consumes frames and executes all callbacks sequentially.

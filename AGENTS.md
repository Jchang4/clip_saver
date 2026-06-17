# AGENTS.md

Clip Saver is a Python package for running Ultralytics YOLO tracking over RTSP streams or video files, converting results to `supervision.Detections`, and handing each processed frame to callbacks.

## Commands

- Install/sync with uv: `uv sync`
- Run tests: `uv run pytest`
- Run a focused test file: `uv run pytest tests/datatypes/test_frame.py`
- Package metadata lives in `pyproject.toml`; `uv.lock` should be updated with `uv lock` after dependency changes.

## Core Flow

`ClipSaver` in `clip_saver/clip_saver.py` is the orchestrator:

1. `BaseVideoSource.get_video_url()` supplies a file path, RTSP URL, or stream-list file.
2. `YOLO(...).track(..., stream=True, persist=True)` yields Ultralytics `Results`.
3. Results are converted with `sv.Detections.from_ultralytics(...)`.
4. Optional detection filters run sequentially with signature `(sv.Detections, list[str]) -> sv.Detections`.
5. A `Frame` dataclass carries the BGR image, detections, timestamp, and source path.
6. Frames are sent through a multiprocessing `Queue` to callbacks.

## Important Modules

- `clip_saver/datatypes/frame.py`: shared frame object; `get_image()` returns a PIL RGB image from the stored OpenCV/BGR array.
- `clip_saver/datatypes/rtsp_url.py`: helper for formatting/parsing the project’s expected RTSP URL shapes.
- `clip_saver/video_source/`: source adapters. `MultiRTSPVideoSource` writes a generated `list.streams` file.
- `clip_saver/callbacks/base.py`: callback lifecycle contract: `start()`, `run(frame)`, `stop()`.
- `clip_saver/callbacks/video_saver.py`: writes optionally annotated video with Supervision.
- `clip_saver/callbacks/tracker.py` and `most_accurate_frame.py`: stateful tracker/class frame collection helpers.

## Testing Guidance

- Use pytest.
- Prefer unit tests for datatypes, URL parsing, filters, and callback state before adding YOLO/video integration tests.
- Avoid tests that require real cameras, RTSP streams, model downloads, or GPU/MPS availability unless explicitly marked as integration tests.
- Keep generated caches and local model files out of git; `.gitignore` already covers `.venv/`, `.pytest_cache/`, `__pycache__/`, `*.pt`, and `list.streams`.

## Current Design Caveats

- Callback multiprocessing is subtle: callback state mutated in the child process is not visible on the original callback object in the parent process.
- `callback.start()` currently runs in the parent process, while `callback.run()` runs in the child process.
- `callback.stop()` exists but is not currently called by `ClipSaver.start()`.
- The `CLIP_SAVER_STOP` environment variable is checked in the main process; setting it inside a child callback process will not reliably stop the parent.

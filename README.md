# Clip Saver

A Python package for processing video streams (RTSP or files) with YOLO object detection and customizable callbacks.

## Installation

```bash
pip install git@github.com:Jchang4/clip_saver.git
```

## Features

- Process RTSP streams or video files with YOLO object detection
- Extensible callback system for custom processing of detection results
- Multiprocessing support for efficient video processing
- Configurable detection filters
- Multiple video sources support

## Quick Start

```python
from clip_saver import ClipSaver
from clip_saver.video_source.rtsp import RTSPVideoSource
from clip_saver.callbacks.info import PrintInfoCallback

# Create a video source
video_source = RTSPVideoSource("rtsp://example.com/stream")

# Initialize ClipSaver
clip_saver = ClipSaver(
    model_path="yolo12x.pt",  # Path to your YOLO model
    video_source=video_source,
    callbacks=[
        PrintInfoCallback(),  # Prints detection information
    ],
    model_kwargs={
        "conf": 0.25,         # Confidence threshold
        "show": True,         # Show detection window
        "device": "mps",      # Device to run on (cpu, cuda, mps)
    },
)

# Start processing
clip_saver.start()

# Stop processing
clip_saver.stop()
```

## Video Sources

### RTSP Stream

```python
from clip_saver.video_source.rtsp import RTSPVideoSource

# Single RTSP stream
video_source = RTSPVideoSource("rtsp://username:password@ip:port/path")

# Or use the RtspUrl helper
from clip_saver.datatypes.rtsp_url import RtspUrl

rtsp_url = RtspUrl(
    username="admin",
    password="password",
    ip_address="192.168.1.100",
    port=554,
    channel=1,
    subtype=0
)
video_source = RTSPVideoSource(str(rtsp_url))
```

### Multiple RTSP Streams

```python
from clip_saver.video_source.rtsp import MultiRTSPVideoSource

video_source = MultiRTSPVideoSource([
    "rtsp://username:password@ip1:port/path",
    "rtsp://username:password@ip2:port/path",
])
```

### Video File

```python
from clip_saver.video_source.file import FileVideoSource

video_source = FileVideoSource("path/to/video.mp4")
```

## Callbacks

Callbacks process each frame with detection results. You can create custom callbacks by extending `BaseCallback`.

### Built-in Callbacks

#### PrintInfoCallback

Prints detection information to the console.

```python
from clip_saver.callbacks.info import PrintInfoCallback

callbacks = [PrintInfoCallback()]
```

#### VideoSaverCallback

Saves processed frames to a video file.

```python
from clip_saver.callbacks.video_saver import VideoSaverCallback

callbacks = [
    VideoSaverCallback(
        output_path="output.mp4",
        skip_no_detections=True,  # Skip frames with no detections
        include_boxes=True        # Draw bounding boxes on output
    )
]
```

#### MostAccurateFrameCallback

Saves the frame with the highest confidence detection.

```python
from clip_saver.callbacks.most_accurate_frame import MostAccurateFrameCallback

# Save the frame (image) with the highest accuracy per class per track_id
callbacks = [MostAccurateFrameCallback()]
```

### Custom Callbacks

Create your own callback by extending the `BaseCallback` class:

```python
from clip_saver.callbacks.base import BaseCallback
from clip_saver.datatypes.frame import Frame

class MyCustomCallback(BaseCallback):
    def start(self):
        # Initialize resources
        pass

    def run(self, frame: Frame):
        # Process the frame
        # frame.image - numpy array with the image
        # frame.detections - supervision Detections object
        # frame.timestamp - ISO formatted timestamp
        # frame.video_path - path to the video source
        pass

    def stop(self):
        # Clean up resources
        pass
```

## Advanced Usage

### Stopping ClipSaver

You can stop ClipSaver in two ways:

1. Call the `stop()` method:

```python
clip_saver.start()  # Starts in the current thread
# In another thread:
clip_saver.stop()
```

2. Set the `CLIP_SAVER_STOP` environment variable:

```python
import os
os.environ["CLIP_SAVER_STOP"] = "true"
```

This is useful for stopping ClipSaver from external processes or callbacks:

```python
import os
from clip_saver.callbacks.base import BaseCallback

class StopSignalCallback(BaseCallback):
    def start(self):
        # Initialize connection to Redis or other signal source
        pass

    def run(self, frame):
        # Check for stop signal
        if self.should_stop():
            os.environ["CLIP_SAVER_STOP"] = "true"

    def stop(self):
        # Clean up
        pass

    def should_stop(self):
        # Check Redis or other signal source
        return False
```

### Detection Filters

You can filter detections using custom filter functions:

```python
def filter_by_class(detections, classnames):
    # Keep only person and car detections
    mask = np.isin(detections.class_id, [classnames.index("person"), classnames.index("car")])
    return detections[mask]

clip_saver = ClipSaver(
    model_path="yolo12x.pt",
    video_source=video_source,
    detections_filter=[filter_by_class],
    callbacks=[PrintInfoCallback()],
)
```

## Security Considerations

When using RTSP URLs with credentials, be careful not to expose them in your code:

```python
# AVOID hardcoding credentials
video_source = RTSPVideoSource("rtsp://admin:password@192.168.1.100:554/stream")

# BETTER: Use environment variables
import os
username = os.environ.get("RTSP_USERNAME")
password = os.environ.get("RTSP_PASSWORD")
video_source = RTSPVideoSource(f"rtsp://{username}:{password}@192.168.1.100:554/stream")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

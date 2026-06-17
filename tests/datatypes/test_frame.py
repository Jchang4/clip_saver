import numpy as np
from PIL import Image

from clip_saver.datatypes.frame import Frame


def test_frame_stores_image_detections_and_metadata():
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    detections = object()

    frame = Frame(
        image=image,
        detections=detections,
        timestamp="2026-06-17T12:00:00",
        video_path="video.mp4",
    )

    assert frame.image is image
    assert frame.detections is detections
    assert frame.timestamp == "2026-06-17T12:00:00"
    assert frame.video_path == "video.mp4"


def test_get_image_returns_rgb_pil_image():
    image = np.array(
        [
            [
                [10, 20, 30],
                [1, 2, 3],
            ]
        ],
        dtype=np.uint8,
    )
    frame = Frame(
        image=image,
        detections=object(),
        timestamp="2026-06-17T12:00:00",
        video_path="video.mp4",
    )

    result = frame.get_image()

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (2, 1)
    np.testing.assert_array_equal(
        np.asarray(result),
        np.array(
            [
                [
                    [30, 20, 10],
                    [3, 2, 1],
                ]
            ],
            dtype=np.uint8,
        ),
    )

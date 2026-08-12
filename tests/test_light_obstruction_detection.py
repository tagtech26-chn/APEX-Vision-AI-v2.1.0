from __future__ import annotations

import cv2
import numpy as np

from app.ai.detection.heuristic import HeuristicDetector
from app.ai.segmentation.heuristic import HeuristicSegmenter


def _room_with_foreground_object() -> np.ndarray:
    image = np.full((300, 400, 3), (90, 125, 165), dtype=np.uint8)
    # Smooth floor-like region in the lower frame.
    image[150:, :] = (105, 135, 170)
    # A large contrasting sofa/object sitting on the floor.
    cv2.rectangle(image, (90, 155), (310, 245), (185, 185, 185), -1)
    cv2.rectangle(image, (110, 140), (290, 185), (175, 175, 175), -1)
    return image


def test_light_detector_finds_foreground_obstruction() -> None:
    image = _room_with_foreground_object()
    detector = HeuristicDetector()

    detections = detector.detect(
        image,
        "sofa. couch. armchair. chair. table. rug. plant. lamp. cabinet. bed. furniture.",
    )

    assert detections
    assert any(
        x1 <= 90 and y1 <= 155 and x2 >= 310 and y2 >= 245
        for x1, y1, x2, y2 in (d.box for d in detections)
    )


def test_light_segmenter_returns_object_masks() -> None:
    image = _room_with_foreground_object()
    segmenter = HeuristicSegmenter()

    masks = segmenter.segment_many(image, boxes=[(90, 140, 310, 245)])

    assert len(masks) == 1
    mask = masks[0]
    assert mask.dtype == np.uint8
    assert int((mask[155:245, 90:310] > 0).sum()) > 0

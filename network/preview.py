"""Build the arena's unmirrored 4:3 preview from the existing camera frame."""
import base64
import math

import cv2
import numpy as np


def arena_preview(frame, sample):
    height, width = frame.shape[:2]
    scale = min(320 / width, 240 / height)
    resized_width = max(1, min(320, round(width * scale)))
    resized_height = max(1, min(240, round(height * scale)))
    left = (320 - resized_width) // 2
    top = (240 - resized_height) // 2
    # Capture/inference are mirrored; the website applies its own mirror.
    original = cv2.flip(frame, 1) if sample.image_mirrored else frame
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        original, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    hands = []
    for hand in sample.hands[:2]:
        if len(hand.landmarks) != 21:
            continue
        points = []
        for x, y, *_ in hand.landmarks:
            if not (math.isfinite(x) and math.isfinite(y)):
                break
            x = 1 - x if sample.image_mirrored else x
            points.append({"x": (left + min(1, max(0, x)) * resized_width) / 320,
                           "y": (top + min(1, max(0, y)) * resized_height) / 240})
        if len(points) == 21:
            hands.append(points)
    payload = {"hands": hands}
    # Busy scenes can exceed the server's size limit at the normal quality.
    for quality in (55, 35, 20):
        ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            break
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        if len(image) <= 48000:
            payload["image"] = image
            break
    return payload

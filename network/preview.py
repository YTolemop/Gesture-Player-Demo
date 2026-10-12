"""Bounded arena preview built from the recognizer's existing capture."""
import base64
import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240


def encode_preview(frame, sample):
    """Return an unmirrored, letterboxed JPEG and aligned 21-point hands.

    The arena mirrors both layers once. Student features and the local preview
    continue to use their original mirrored capture.
    """
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    mirrored = sample.image_mirrored
    source = cv2.flip(frame, 1) if mirrored else frame
    scale = min(WIDTH / width, HEIGHT / height)
    resized_width = max(1, min(WIDTH, round(width * scale)))
    resized_height = max(1, min(HEIGHT, round(height * scale)))
    offset_x = (WIDTH - resized_width) // 2
    offset_y = (HEIGHT - resized_height) // 2
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = cv2.resize(
        source, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    hands = []
    for hand in (sample.left, sample.right):
        if hand is None:
            continue
        points = np.asarray(hand)
        if points.shape != (21, 3) or not np.isfinite(points[:, :2]).all():
            continue
        x = np.clip(points[:, 0], 0, 1)
        y = np.clip(points[:, 1], 0, 1)
        if mirrored:
            x = 1 - x
        hands.append([{'x': float((px * resized_width + offset_x) / WIDTH),
                       'y': float((py * resized_height + offset_y) / HEIGHT)}
                      for px, py in zip(x, y)])
    ok, jpeg = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 55])
    if not ok:
        return None
    image = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')
    if len(image) > 48000:
        return {'hands': hands}
    return {'image': image, 'hands': hands}

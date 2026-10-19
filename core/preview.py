"""Diagnostic drawing only; labels are raw model output, not student decisions."""
import cv2

CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))


def show_preview(frame, sample):
    """Draw on a copy so recordings and inference keep their original pixels."""
    canvas = frame.copy()
    for index, hand in enumerate(sample.hands):
        points = [(round(x * sample.width), round(y * sample.height))
                  for x, y, z in hand.landmarks]
        for a, b in CONNECTIONS:
            cv2.line(canvas, points[a], points[b], (0, 220, 0), 2)
        for point in points:
            cv2.circle(canvas, point, 3, (0, 100, 255), -1)
        categories = ", ".join(f"{c['category_name']}:{c['score']:.2f}"
                               for c in hand.handedness)
        cv2.putText(canvas, f"hand[{index}] raw {categories}", (10, 55 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    cv2.putText(canvas, f"hands={len(sample.hands)} | Q/Esc: stop", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
    cv2.imshow("MediaPipe raw landmarks", canvas)
    key = cv2.waitKey(1) & 0xff
    return key not in (27, ord("q")) and cv2.getWindowProperty(
        "MediaPipe raw landmarks", cv2.WND_PROP_VISIBLE) >= 1

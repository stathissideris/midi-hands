import math
from pathlib import Path

import cv2
import mediapipe as mp
import mido
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
PORT_NAME = "midi-hands"

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8

CC_WRIST_Y = 1     # mod wheel
CC_PINCH = 11      # expression

CHANNEL_BY_HAND = {"Left": 0, "Right": 1}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for landmarks in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def to_cc(value, lo, hi):
    if hi <= lo:
        return 0
    norm = (value - lo) / (hi - lo)
    return max(0, min(127, int(round(norm * 127))))


class CCSender:
    """Sends MIDI CC, deduping unchanged values per (channel, cc)."""

    def __init__(self, port):
        self.port = port
        self.last = {}

    def send(self, channel, cc, value):
        key = (channel, cc)
        if self.last.get(key) == value:
            return
        self.last[key] = value
        self.port.send(mido.Message("control_change", channel=channel, control=cc, value=value))


def main():
    base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    port = mido.open_output(PORT_NAME, virtual=True)
    print(f"Opened virtual MIDI port: {PORT_NAME}")
    sender = CCSender(port)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or int(
                cv2.getTickCount() * 1000 / cv2.getTickFrequency()
            )
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.hand_landmarks:
                draw(frame, result.hand_landmarks)

                for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                    # Mediapipe reports handedness from the camera's POV; we flipped
                    # the frame, so swap to match the user's actual hand.
                    raw = handed[0].category_name
                    label = "Right" if raw == "Left" else "Left"
                    channel = CHANNEL_BY_HAND.get(label, 0)

                    wrist_y = 1.0 - landmarks[WRIST].y  # invert: hand up = high value
                    sender.send(channel, CC_WRIST_Y, to_cc(wrist_y, 0.0, 1.0))

                    dx = landmarks[THUMB_TIP].x - landmarks[INDEX_TIP].x
                    dy = landmarks[THUMB_TIP].y - landmarks[INDEX_TIP].y
                    pinch = math.hypot(dx, dy)
                    # ~0.02 = touching, ~0.25 = wide open
                    sender.send(channel, CC_PINCH, to_cc(pinch, 0.02, 0.25))

            cv2.imshow("MediaPipe Hands - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    port.close()


if __name__ == "__main__":
    main()

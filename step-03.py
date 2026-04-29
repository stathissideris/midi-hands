"""Step 3 — note on/off from finger taps.

Builds on step-02 (wrist-Y → CC1, thumb–index pinch → CC11, per-hand channel)
and adds note triggering on the *right* hand: thumb tip touching another
fingertip sends note_on; pulling away sends note_off. Each finger is a
different note. We use hysteresis (separate on/off thresholds) and a small
state machine so each event is sent only once per gesture.
"""

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
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

CC_WRIST_Y = 1
CC_PINCH = 11

CHANNEL_BY_HAND = {"Left": 0, "Right": 1}

# Right-hand fingertip → MIDI note. C major pentatonic subset (C E G A) —
# no semitones or leading tones, so any combination sounds consonant.
FINGER_NOTES = {
    INDEX_TIP: 60,   # C4
    MIDDLE_TIP: 64,  # E4
    RING_TIP: 67,    # G4
    PINKY_TIP: 69,   # A4
}

# Hysteresis thresholds in normalized image coords. The gap between the two
# prevents the note from flapping on/off when the distance hovers at the edge.
NOTE_ON_DIST = 0.06
NOTE_OFF_DIST = 0.06
NOTE_VELOCITY = 100

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


def draw_thresholds(frame, thumb, on_dist, off_dist):
    """Show the note-on and note-off boundaries around the thumb tip.

    Distance is hypot of *normalized* (x, y) deltas, so when projected to
    pixels the constant-distance locus is an ellipse with axes (W*r, H*r),
    not a circle (the image isn't square).
    """
    h, w = frame.shape[:2]
    cx, cy = int(thumb.x * w), int(thumb.y * h)
    cv2.ellipse(frame, (cx, cy),
                (int(off_dist * w), int(off_dist * h)),
                0, 0, 360, (0, 165, 255), 1)     # orange = note_off threshold
    cv2.ellipse(frame, (cx, cy),
                (int(on_dist * w), int(on_dist * h)),
                0, 0, 360, (255, 0, 0), 1)       # blue = note_on threshold


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


class NoteTrigger:
    """Per-finger note on/off state machine.

    For each tracked finger, we remember whether its note is currently sounding.
    A note_on fires the first frame the thumb–finger distance crosses below
    `on_dist`; a note_off fires the first frame it crosses back above
    `off_dist`. Between those two thresholds the state is held — that's the
    hysteresis that keeps notes from chattering.
    """

    def __init__(self, port, channel, finger_notes, on_dist, off_dist, velocity):
        self.port = port
        self.channel = channel
        self.finger_notes = finger_notes
        self.on_dist = on_dist
        self.off_dist = off_dist
        self.velocity = velocity
        self.active = set()  # fingertip landmark indices currently sounding

    def update(self, landmarks):
        thumb = landmarks[THUMB_TIP]
        for finger, note in self.finger_notes.items():
            tip = landmarks[finger]
            d = math.hypot(thumb.x - tip.x, thumb.y - tip.y)
            if finger in self.active:
                if d > self.off_dist:
                    self.active.discard(finger)
                    self.port.send(mido.Message(
                        "note_off", channel=self.channel, note=note, velocity=0))
            else:
                if d < self.on_dist:
                    self.active.add(finger)
                    self.port.send(mido.Message(
                        "note_on", channel=self.channel, note=note, velocity=self.velocity))

    def all_off(self):
        for finger in list(self.active):
            note = self.finger_notes[finger]
            self.port.send(mido.Message(
                "note_off", channel=self.channel, note=note, velocity=0))
            self.active.discard(finger)


def open_midi_port(name):
    try:
        port = mido.open_output(name, virtual=True)
        print(f"Opened virtual MIDI port: {name}")
        return port
    except (NotImplementedError, RuntimeError):
        pass

    available = mido.get_output_names()
    match = next((n for n in available if name.lower() in n.lower()), None)
    if match is None:
        raise RuntimeError(
            f"Could not open virtual MIDI port '{name}'. On Windows, install a "
            f"loopback driver (e.g. loopMIDI) and create a port named '{name}', "
            f"then re-run. Available ports: {available}"
        )
    port = mido.open_output(match)
    print(f"Opened existing MIDI port: {match}")
    return port


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

    port = open_midi_port(PORT_NAME)
    sender = CCSender(port)
    trigger = NoteTrigger(
        port,
        channel=CHANNEL_BY_HAND["Right"],
        finger_notes=FINGER_NOTES,
        on_dist=NOTE_ON_DIST,
        off_dist=NOTE_OFF_DIST,
        velocity=NOTE_VELOCITY,
    )

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

            right_seen = False
            if result.hand_landmarks:
                draw(frame, result.hand_landmarks)

                for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                    raw = handed[0].category_name
                    label = "Right" if raw == "Left" else "Left"
                    channel = CHANNEL_BY_HAND.get(label, 0)

                    wrist_y = 1.0 - landmarks[WRIST].y
                    sender.send(channel, CC_WRIST_Y, to_cc(wrist_y, 0.0, 1.0))

                    dx = landmarks[THUMB_TIP].x - landmarks[INDEX_TIP].x
                    dy = landmarks[THUMB_TIP].y - landmarks[INDEX_TIP].y
                    pinch = math.hypot(dx, dy)
                    sender.send(channel, CC_PINCH, to_cc(pinch, 0.02, 0.25))

                    if label == "Right":
                        right_seen = True
                        trigger.update(landmarks)
                        draw_thresholds(frame, landmarks[THUMB_TIP],
                                        NOTE_ON_DIST, NOTE_OFF_DIST)

            # If the right hand vanished mid-tap, release any held notes so
            # they don't get stuck on in the DAW.
            if not right_seen:
                trigger.all_off()

            cv2.imshow("MediaPipe Hands - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    trigger.all_off()
    cap.release()
    cv2.destroyAllWindows()
    port.close()


if __name__ == "__main__":
    main()

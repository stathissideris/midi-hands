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
    # hand_landmarks_list is a list with one entry per detected hand. Each
    # entry is itself a list of 21 NormalizedLandmark objects with .x, .y
    # and .z attributes. The coordinates are normalised to 0..1 (relative to
    # image width and height), so a landmark at the very centre of the frame
    # would have x=0.5, y=0.5. The full landmark list and what each index
    # corresponds to (wrist, thumb tip, …) is documented at:
    # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    # Roughly:
    #
    #   [
    #       [  # first hand — 21 landmarks
    #           NormalizedLandmark(x=0.52, y=0.71, z= 0.00),  #  0: wrist
    #           NormalizedLandmark(x=0.48, y=0.65, z=-0.01),  #  1: thumb CMC
    #           NormalizedLandmark(x=0.45, y=0.58, z=-0.02),  #  2: thumb MCP
    #           ...
    #           NormalizedLandmark(x=0.55, y=0.30, z=-0.05),  # 20: pinky tip
    #       ],
    #       [  # second hand — another 21 landmarks
    #           ...
    #       ],
    #   ]

    # frame.shape is (height, width, channels); we need width and height to
    # turn the normalised 0..1 coordinates back into actual pixel positions.
    h, w = frame.shape[:2]

    # Loop over each detected hand independently — we draw one skeleton per
    # hand.
    for landmarks in hand_landmarks_list:
        # Convert every landmark's normalised (x, y) into integer pixel
        # coordinates. We ignore z here because we're drawing onto a 2D
        # image; z would only matter for 3D rendering or depth-based logic.
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Draw the green skeleton. HAND_CONNECTIONS is a list of (a, b)
        # index pairs: each pair says "draw a line between landmark a and
        # landmark b" (e.g. (0, 1) connects the wrist to the base of the
        # thumb). Note that OpenCV uses BGR colour order, so (0, 255, 0)
        # is green.
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

        # Draw a filled red circle at each of the 21 joint positions on top
        # of the skeleton so the joints stand out. Again, BGR — (0, 0, 255)
        # is red. The -1 thickness means "fill the circle".
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


def detect_hands(frame, cap, landmarker):
    # OpenCV gives us frames in BGR colour order, but MediaPipe expects RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the raw pixel array in MediaPipe's own Image type so the detector
    # can consume it.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # MediaPipe's video mode needs a monotonically increasing timestamp in
    # *milliseconds*. The preferred source is the webcam's own clock,
    # exposed by OpenCV as CAP_PROP_POS_MSEC. Some webcams report 0 for that
    # property, in which case we have to make our own clock instead:
    #   - cv2.getTickCount()     → a raw CPU tick counter (ticks since boot
    #                              or some reference event). Its unit is
    #                              "ticks", not seconds, and the size of a
    #                              tick varies from machine to machine.
    #   - cv2.getTickFrequency() → how many ticks happen per second on this
    #                              machine.
    # Dividing one by the other gives seconds; multiplying by 1000 turns
    # that into milliseconds, which is what MediaPipe wants.
    ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or int(
        cv2.getTickCount() * 1000 / cv2.getTickFrequency()
    )

    # Run hand detection. result.hand_landmarks is a list with one entry per
    # detected hand; each entry is 21 (x, y, z) landmarks (wrist, knuckles,
    # fingertips) in normalised 0..1 coordinates.
    result = landmarker.detect_for_video(mp_image, ts_ms)

    return result


def distance(a, b):
    # 2D Euclidean distance between two landmarks. We ignore the .z
    # component on purpose — for the gestures we care about (pinch,
    # finger taps), the screen-plane distance is what matters and z
    # is noisier. math.hypot(dx, dy) is the same as sqrt(dx*dx + dy*dy)
    # but a touch more numerically stable.
    #
    # Inputs are NormalizedLandmark objects with .x and .y in 0..1, so
    # the returned distance is also in normalised image-space units
    # (i.e. ~0.0 means the two points are touching, ~1.4 would be the
    # diagonal of the entire frame).
    return math.hypot(a.x - b.x, a.y - b.y)


def to_cc(value, lo, hi):
    # Map a continuous `value` from the range [lo, hi] onto a MIDI CC value
    # in the range 0..127. MIDI CCs are 7-bit integers, so 127 is the
    # highest value any controller can send. For example, with lo=0.0 and
    # hi=1.0:
    #   to_cc(0.0, 0.0, 1.0)  ->   0
    #   to_cc(0.5, 0.0, 1.0)  ->  64   (rounded from 63.5)
    #   to_cc(1.0, 0.0, 1.0)  -> 127
    #   to_cc(2.0, 0.0, 1.0)  -> 127   (clamped — value above hi)
    #   to_cc(-1.0, 0.0, 1.0) ->   0   (clamped — value below lo)

    # Defensive guard: if the caller passed a degenerate range (hi not
    # strictly greater than lo) we'd divide by zero or get a negative span,
    # so just return 0 instead of crashing.
    if hi <= lo:
        return 0

    # Normalise `value` into a 0..1 fraction within the [lo, hi] range.
    # If value == lo this is 0; if value == hi this is 1; values outside
    # the range produce numbers below 0 or above 1, which we clamp below.
    norm = (value - lo) / (hi - lo)

    # Scale the 0..1 fraction up to 0..127, round to the nearest integer,
    # then clamp into the valid MIDI CC range. The min/max pair is what
    # turns out-of-range inputs into a hard 0 or 127 rather than letting
    # them produce illegal MIDI values.
    return max(0, min(127, int(round(norm * 127))))


class CCSender:
    """Sends MIDI Control Change messages, deduping unchanged values.

    A "Control Change" (CC) is one of the most common MIDI messages:
    it tells a synth or DAW to change the value of a numbered controller
    (CC 1 is the mod wheel, CC 7 is volume, CC 11 is expression, etc.).
    Each message carries three numbers:
      - channel: which of the 16 MIDI channels the message is for
                 (we use channel 0 for the left hand, 1 for the right)
      - cc:      which controller number is being changed (0..127)
      - value:   the new value for that controller (0..127)

    The webcam runs at ~30 frames per second, so the loop in main() will
    call to_cc() and try to send a CC value 30 times a second per
    controller. Most of those frames will produce *the same* value as the
    previous frame (your hand isn't really moving every 33 ms), and
    re-sending identical CC values is wasteful: it floods the MIDI bus,
    can cause hiccups on cheap hardware, and bloats automation lanes if
    the DAW is recording. So this class remembers the last value sent
    per (channel, cc) pair and only forwards a new message when the
    value actually changes.
    """

    def __init__(self, port):
        # The mido output port we'll write MIDI messages to. Could be a
        # virtual port created with mido.open_output(..., virtual=True),
        # or an existing port (e.g. loopMIDI on Windows).
        self.port = port

        # Cache of the last value we sent for each (channel, cc) pair.
        # Shape: {(channel, cc): last_value}. A different (channel, cc)
        # is a totally separate stream, so each one needs its own slot.
        self.last = {}

    def send(self, channel, cc, value):
        # Build the lookup key for this controller stream. Two different
        # CC numbers, or the same CC on two different channels, are
        # independent — they each get their own dedupe slot.
        key = (channel, cc)

        # Skip the send entirely if we already pushed this exact value
        # for this controller last time. dict.get returns None if the key
        # has never been seen, and None != value (assuming `value` is an
        # int), so the very first call always passes through.
        if self.last.get(key) == value:
            return

        # Remember what we're about to send so the next call can dedupe.
        self.last[key] = value

        # Actually emit the MIDI message. mido takes care of packing the
        # status byte and two data bytes; "control_change" maps to the
        # standard MIDI CC message type.
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
            d = distance(thumb, tip)
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
    # first port whose name contains midi-hands, or None if there isn't one.
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
        min_tracking_confidence=0.7,
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

            # Mirror the frame horizontally so the preview behaves like a
            # real mirror — your right hand appears on the right side of the
            # screen. We do this before detection so MediaPipe's left/right
            # hand labels match the user's perspective.
            frame = cv2.flip(frame, 1)

            result = detect_hands(frame, cap, landmarker)

            right_seen = False
            if result.hand_landmarks:
                draw(frame, result.hand_landmarks)

                for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                    # result.handedness is a list parallel to result.hand_landmarks:
                    # one entry per detected hand, in the same order. Each entry
                    # is itself a list of Category objects ranked by confidence;
                    # we only ever read [0], the top classification. A Category
                    # has .category_name ("Left" or "Right"), .score (0..1
                    # confidence), .index, and .display_name. Roughly:
                    #
                    #   [
                    #       [Category(category_name="Left",  score=0.97, ...)],
                    #       [Category(category_name="Right", score=0.93, ...)],
                    #   ]
                    #
                    # Docs:
                    # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
                    #
                    # Mediapipe reports handedness from the camera's POV; we
                    # flipped the frame, so swap to match the user's actual
                    # hand.
                    raw = handed[0].category_name
                    label = "Right" if raw == "Left" else "Left"

                    # The left hand drives the continuous CCs (wrist height
                    # → mod wheel, thumb–index pinch → expression). The
                    # right hand is reserved for note triggering below.
                    if label == "Left":
                        channel = CHANNEL_BY_HAND[label]

                        wrist_y = 1.0 - landmarks[WRIST].y  # invert: hand up = high value
                        sender.send(channel, CC_WRIST_Y, to_cc(wrist_y, 0.0, 1.0))

                        pinch = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
                        # ~0.02 = touching, ~0.25 = wide open
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

            cv2.imshow("MIDI Hands - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    trigger.all_off()
    cap.release()
    cv2.destroyAllWindows()
    port.close()


if __name__ == "__main__":
    main()

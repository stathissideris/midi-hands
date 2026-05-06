from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

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

            if result.hand_landmarks:
                draw(frame, result.hand_landmarks)

            cv2.imshow("MediaPipe Hands - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

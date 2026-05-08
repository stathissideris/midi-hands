import cv2


def main():
    # cv2.VideoCapture(0) opens the default webcam. The integer is the
    # device index — 0 is the first camera the OS knows about, 1 the
    # second, and so on. On a laptop with a built-in camera that's
    # usually the one you get.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # The main loop: grab a frame, draw on it, show it, repeat. This runs
    # as fast as the webcam can deliver frames (usually ~30 fps).
    while cap.isOpened():
        # cap.read() returns a (success_flag, frame) tuple. `frame` is a
        # NumPy array of shape (height, width, 3) holding the pixel data
        # in BGR order. If the read fails (e.g. a transient hiccup from
        # the camera) we just skip this iteration and try again.
        ok, frame = cap.read()
        if not ok:
            continue

        # Mirror the frame horizontally so the preview behaves like a
        # real mirror — moving your hand to the right moves it right on
        # screen too. The second argument is the flip code: 1 means flip
        # around the vertical axis (left/right), 0 would flip top/bottom.
        frame = cv2.flip(frame, 1)

        # Draw the text "MIDI hands" in the top-left corner of the frame.
        # cv2.putText draws *onto* the frame in place — it doesn't return
        # a new image. Arguments, in order:
        #   - frame:        the image to draw on
        #   - text:         the string to render
        #   - origin:       (x, y) pixel position of the text's bottom-left
        #                   corner. y grows downward in image coordinates,
        #                   so (10, 30) is near the top-left.
        #   - font face:    one of cv2's built-in fonts
        #   - font scale:   multiplier on the font's base size
        #   - colour:       BGR tuple — (0, 255, 0) is green
        #   - thickness:    stroke width in pixels
        #   - line type:    cv2.LINE_AA enables anti-aliasing for smoother
        #                   edges (without it the text looks jaggy)
        cv2.putText(
            frame,
            "MIDI hands",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Push the frame to a window. The first argument is the window
        # title; OpenCV creates the window the first time it sees a new
        # title and reuses it on subsequent calls.
        cv2.imshow("MIDI Hands - press q to quit", frame)

        # cv2.waitKey(1) waits up to 1 ms for a key press and returns the
        # key code (or -1 if no key was pressed). The & 0xFF masks off
        # the high bits so we get a plain ASCII code we can compare with
        # ord("q"). Without the waitKey call, imshow wouldn't actually
        # render anything — it's also what gives the GUI a chance to
        # process events.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Tidy up: hand the camera back to the OS and close the window.
    # Without these, the camera can stay locked until the Python process
    # exits, and stray windows can linger.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

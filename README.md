# midi-hands

Turn hand gestures into MIDI CC messages using your webcam.

Built around Google's [MediaPipe](https://ai.google.dev/edge/mediapipe) Hand
Landmarker, OpenCV for video capture, and [`mido`](https://mido.readthedocs.io/)
for MIDI output. The script opens a virtual MIDI port that any DAW (Logic,
Ableton, Reaper, VCV Rack, …) will see immediately — no IAC bus setup required.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) and Python 3.12.

```sh
uv sync                 # install dependencies
./download-models.sh    # fetch the MediaPipe hand_landmarker.task model
```

## Run

```sh
uv run main.py
```

Then, in your DAW, select **`midi-hands`** as a MIDI input on an armed track.
Press `q` in the preview window to quit.

### Default mappings

| Gesture                          | MIDI                                    |
|----------------------------------|-----------------------------------------|
| Left hand                        | Channel 1                               |
| Right hand                       | Channel 2                               |
| Wrist height (Y)                 | CC 1 (mod wheel) — hand up = higher     |
| Thumb–index pinch distance       | CC 11 (expression) — closed = 0         |
| Right thumb → index touch        | Note on/off C4 (60)                     |
| Right thumb → middle touch       | Note on/off E4 (64)                     |
| Right thumb → ring touch         | Note on/off G4 (67)                     |
| Right thumb → pinky touch        | Note on/off A4 (69)                     |

Unchanged CC values are not re-sent, so the bus stays quiet when you hold still.
Note triggering uses hysteresis (separate on/off thresholds) so a steady touch
won't chatter, and an `all-notes-off` is sent if the right hand leaves the
frame to prevent stuck notes.

## Repository layout

This project doubles as teaching material. Alongside the final script, there
is a series of progressively-built `step-XX.py` files that introduce one new
concept at a time.

| File          | What it adds                                                |
|---------------|-------------------------------------------------------------|
| `step-01.py`  | MediaPipe hand detection on the webcam, drawing landmarks   |
| `step-02.py`  | Adds a virtual MIDI port and sends CCs from the wrist       |
| `step-03.py`  | Note on/off from finger taps on the right hand, with a state machine and hysteresis thresholds |
| `…`           | Further refinements (smoothing, more gestures, …)           |
| `main.py`     | The final, presentable version                              |

The step files are standalone — each one runs on its own with `uv run step-XX.py`.

## Files

- `main.py` — the finished script
- `step-XX.py` — incremental teaching steps
- `download-models.sh` — fetches the `hand_landmarker.task` bundle
- `hand_landmarker.task` — MediaPipe model bundle (gitignored, downloaded on demand)

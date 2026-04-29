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

Unchanged values are not re-sent, so the bus stays quiet when you hold still.

## Repository layout

This project doubles as teaching material. Alongside the final script, there
is a series of progressively-built `step-XX.py` files that introduce one new
concept at a time.

| File          | What it adds                                                |
|---------------|-------------------------------------------------------------|
| `step-01.py`  | MediaPipe hand detection on the webcam, drawing landmarks   |
| `step-02.py`  | Adds a virtual MIDI port and sends CCs from the wrist       |
| `…`           | Further refinements (smoothing, more gestures, …)           |
| `main.py`     | The final, presentable version                              |

The step files are standalone — each one runs on its own with `uv run step-XX.py`.

## Files

- `main.py` — the finished script
- `step-XX.py` — incremental teaching steps
- `download-models.sh` — fetches the `hand_landmarker.task` bundle
- `hand_landmarker.task` — MediaPipe model bundle (gitignored, downloaded on demand)

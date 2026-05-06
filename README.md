# midi-hands

Turn hand gestures into MIDI CC messages using your webcam.

Built around Google's [MediaPipe](https://ai.google.dev/edge/mediapipe) Hand
Landmarker, OpenCV for video capture, and [`mido`](https://mido.readthedocs.io/)
for MIDI output. The script opens a virtual MIDI port that any DAW (Logic,
Ableton, Reaper, VCV Rack, …) will see immediately — no IAC bus setup required.

## How it works

```mermaid
flowchart LR
    hands["Your hands"] -->|move| webcam["Webcam"]
    webcam -->|frames| opencv["OpenCV"]
    opencv -->|RGB image| mediapipe["MediaPipe"]
    mediapipe -->|21 landmarks| math["Math"]
    math -->|CC &amp; note events| mido["mido"]
    mido -->|MIDI messages| daw["Your DAW"]
    daw -->|sound| speakers["Speakers"]
    speakers -->|sound waves| ears["Your ears"]
```

Each frame from the webcam is read by OpenCV and handed to MediaPipe, which
returns the 3D position of 21 points per hand (wrist, knuckles, fingertips).  A
bit of math turns those positions into useful musical signals — wrist height
becomes a CC value, the thumb-to-finger distance becomes another, and finger-tap
detection becomes note on/off messages. `mido` sends those out over a virtual
MIDI port your DAW listens to.

## Setup

This project is a small Python program. You don't need to know anything about
Python — a tool called [`uv`](https://github.com/astral-sh/uv) handles
installing Python and all the libraries for you. Pick the section below that
matches your operating system and follow it once. After that, jump to
[Run](#run).

### Mac

1. **Open the Terminal.** Press `Cmd`+`Space`, type `Terminal`, press Enter.
   A window with a text prompt will open. You'll be typing commands here. To
   "run" a command, copy it, paste it into Terminal, and press Enter.

2. **Install `uv`.** Paste this command and press Enter:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   When it finishes, **close the Terminal window and open a new one** so the
   change takes effect.

3. **Download the project.** On
   [the GitHub page](https://github.com/) for this repo, click the green
   **Code** button and choose **Download ZIP**. Unzip the file (double-click
   it). You'll get a folder called `midi-hands-master` (or similar). Move it
   somewhere you'll remember, like your Desktop.

4. **Go into the project folder in the Terminal.** Type `cd ` (with a space),
   then drag the project folder from Finder onto the Terminal window — its
   path will be filled in for you. Press Enter.

5. **Install everything.** Run these two commands one at a time:

   ```sh
   uv sync
   ./download-models.sh
   ```

   The first one downloads Python and the libraries the project needs. The
   second one downloads the hand-detection model. Both are one-time steps.

### Windows

1. **Install loopMIDI** (a free tool that creates virtual MIDI cables, which
   Windows doesn't have built in). Download it from
   [tobias-erichsen.de/software/loopmidi.html](https://www.tobias-erichsen.de/software/loopmidi.html)
   and run the installer. Open loopMIDI, type `midi-hands` in the box at the
   bottom-left, and click the **+** button. Leave loopMIDI running in the
   background whenever you use this project.

2. **Open PowerShell.** Press the Windows key, type `PowerShell`, press Enter.
   A blue window with a text prompt will open. You'll be typing commands
   here. To "run" a command, copy it, paste it in (right-click pastes), and
   press Enter.

3. **Install `uv`.** Paste this command and press Enter:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   When it finishes, **close the PowerShell window and open a new one** so
   the change takes effect.

4. **Download the project.** On
   [the GitHub page](https://github.com/) for this repo, click the green
   **Code** button and choose **Download ZIP**. Right-click the downloaded
   file and choose **Extract All…**. Move the resulting folder somewhere
   you'll remember, like your Desktop.

5. **Go into the project folder in PowerShell.** Type `cd ` (with a space),
   then drag the project folder from File Explorer onto the PowerShell
   window — its path will be filled in for you. Press Enter.

6. **Install everything.** Run these commands one at a time:

   ```powershell
   uv sync
   .\download-models.ps1
   ```

   The first one downloads Python and the libraries the project needs. The
   second one downloads the hand-detection model. Both are one-time steps.

   If PowerShell refuses to run `.\download-models.ps1` with a message about
   scripts being disabled, run this once and try again:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

### Linux

1. **Open a terminal.** On most distributions you can press `Ctrl`+`Alt`+`T`,
   or find "Terminal" in your applications menu.

2. **Install `uv`.** Paste this command and press Enter:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   When it finishes, **close the terminal and open a new one** so the change
   takes effect.

3. **Download the project.** On
   [the GitHub page](https://github.com/) for this repo, click the green
   **Code** button and choose **Download ZIP**. Extract the archive (your
   file manager can usually do this with a right-click) and move the folder
   somewhere you'll remember, like your home directory.

4. **Go into the project folder in the terminal.** Use `cd` followed by the
   path, e.g.:

   ```sh
   cd ~/midi-hands-master
   ```

5. **Install everything.** Run these two commands one at a time:

   ```sh
   uv sync
   ./download-models.sh
   ```

   The first one downloads Python and the libraries the project needs. The
   second one downloads the hand-detection model. Both are one-time steps.

   You may also need a few system packages for the camera and MIDI to work
   (most distributions already have them). On Debian/Ubuntu, for example:

   ```sh
   sudo apt install libgl1 libglib2.0-0 librtmidi-dev
   ```

## Run

Open a terminal (or PowerShell on Windows), navigate to the project folder
the same way you did during setup, and run:

```sh
uv run main.py
```

A small window showing your webcam will open. Then, in your DAW, select
**`midi-hands`** as a MIDI input on an armed track. Press `q` in the preview
window to quit.

> **Windows note:** make sure loopMIDI is running and has a port called
> `midi-hands` (see step 1 of the Windows setup above), otherwise the script
> won't be able to send MIDI.

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
An `all-notes-off` is sent if the right hand leaves the frame to prevent stuck
notes.

## Repository layout

This project doubles as teaching material. Alongside the final script, there
is a series of progressively-built `step-XX.py` files that introduce one new
concept at a time.

| File          | What it adds                                                |
|---------------|-------------------------------------------------------------|
| `step-01.py`  | MediaPipe hand detection on the webcam, drawing landmarks   |
| `step-02.py`  | Adds a virtual MIDI port and sends CCs from the wrist       |
| `step-03.py`  | Note on/off from finger taps on the right hand, with a state machine |
| `main.py`     | The final, presentable version                              |

The step files are standalone — each one runs on its own with `uv run step-XX.py`.

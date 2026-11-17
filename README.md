# SmartTrace

SmartTrace is a Python/OpenCV webcam prototype that detects foreground motion, tracks detected regions across frames, and saves timestamped image evidence when configurable behavioral rules are met. It is intended as a demonstration project; it is not a replacement for a validated security system.

## Problem and approach

Basic motion alarms can be triggered by brief movement, shadows, pets, or lighting changes, and saving every movement can create unnecessary evidence files. SmartTrace demonstrates how classical computer vision and simple behavioral rules can filter some brief events before saving a snapshot. It uses foreground motion and contour regions rather than recognizing or classifying people or other objects.

## Project documentation

See the [CSC 126 project documentation](<docs/CSC 126 - Documentation.pdf>) for the project objectives, techniques, screenshots, behavior, limitations, and conclusion.

## Features

- Motion segmentation using MOG2 background subtraction.
- Grayscale conversion, Gaussian blur, thresholding, and morphological cleanup.
- Contour-based detection with lightweight center-point tracking.
- Live bounding boxes, tracking IDs, status text, and timestamps.
- Evidence snapshots grouped by trigger type under `snapshots/`.

## Limitations

- Motion regions are not semantic objects: the application cannot tell a person from a pet, shadow, or other moving pixels.
- Lighting changes and camera noise can affect foreground detection and stationary timing.
- Center-point tracking can confuse or merge nearby objects, especially when they overlap.
- Loitering uses displacement from an object's starting position; the current code does not implement a separate frame-velocity threshold.

## Alert rules

The current thresholds are implemented in `main.py`:

| Trigger | Rule | Output folder |
| --- | --- | --- |
| Loitering | A tracked object remains within 120 pixels of its starting position for at least 5 seconds. | `snapshots/loitering/` |
| Lingering | A tracked object has been tracked for at least 20 seconds. | `snapshots/lingering/` |
| Frequency | A tracked object has more than 3 appearances. A reappearance is counted once after one or more missed frames while the track is retained (up to 2 seconds). | `snapshots/frequency/` |

Loitering is checked before lingering, and lingering before frequency. Each rule saves at most one snapshot per application run.

## Requirements

- Python 3.11 or newer
- A webcam available as camera device 0
- The Python packages listed in `requirements.txt`

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run the commands below using the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

1. Connect a webcam and allow camera access for your terminal or Python.
2. Open a terminal in the project directory.
3. Activate the virtual environment if it is not already active.
4. Start the application:

   ```powershell
   python main.py
   ```

5. The live feed opens fullscreen. Press **q** to close it. Captured evidence is written to the corresponding `snapshots/` subfolder.

The application currently reads from the default webcam (`cv2.VideoCapture(0)`). There is no prerecorded sample input video in this repository. Existing files in `snapshots/` are example output evidence images, organized by trigger:

- [Loitering example](snapshots/loitering/loitering_snapshot_20261111_131458.jpg)
- [Lingering example](snapshots/lingering/lingering_snapshot_20261111_111004.jpg)
- [Frequency example](snapshots/frequency/frequency_snapshot_20261111_110940.jpg)

New snapshots use timestamped filenames and are saved into the same folders. Ensure the application has permission to write to the project directory.

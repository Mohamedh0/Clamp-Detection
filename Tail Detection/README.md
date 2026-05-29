# Tail Clamp Detection

A YOLOv11-based computer vision system that detects **tail clamps** or **internal head of clamps** in video footage, measures distances between them, and produces annotated output videos.

**[Output Videos](https://drive.google.com/drive/folders/1ZrzVRBUjZldc0StWBcC0B9_5H5b9Hen4?usp=sharing)**
---

## 📁 Project Structure

```
clamp_detection/
├── notebooks/
│   └── tail_clamp_detection.ipynb   # Original exploration notebook
└── src/
    ├── train.py                     # Dataset download & model training
    ├── detector.py                  # Core detection & annotation logic
    └── run.py                       # Entry point for batch video processing
```

---

## ⚙️ Installation

```bash
pip install ultralytics roboflow opencv-python numpy
```

---

## 🚀 Usage

### 1. Train the Model

Download the dataset from Roboflow and train a YOLOv11m model:

```bash
python src/train.py
```

> **Note:** Replace `YOUR_API_KEY` in `train.py` with your actual Roboflow API key before running.

Training artifacts (weights, logs) are saved under:
```
runs/detect/Tail-clamp-detection-2/train/
```

---

### 2. Run Detection on Videos

Edit the paths in `src/run.py`:

```python
WEIGHTS   = "runs/detect/Tail-clamp-detection-2/train/weights/best.pt"
CLIPS_DIR = "/path/to/your/clips"
```

Then run:

```bash
python src/run.py
```

Each `.mp4` in `CLIPS_DIR` is processed and saved as `output_<original_name>.mp4`.

---

### 3. Use the Detector as a Module

You can import `detector.py` into your own scripts:

```python
from detector import load_model, save_video_detection

model = load_model("path/to/best.pt")
save_video_detection(model, "input.mp4", "output_result")
```

---

## 🧠 How It Works

Each frame is passed through the YOLOv11 model, which detects two classes:

| Class ID | Label  |
|----------|--------|
| `0`      | Clamp  |
| `1`      | Tail   |

The following measurements are computed per frame and drawn on the annotated video:

- **Short–Long Tail distance** — Euclidean distance between the shortest and longest detected tail (based on bounding box diagonal), drawn in.
- **Short Tail–Nearest Clamp distance** — Euclidean distance from the shortest tail to the closest clamp, drawn in.

A HUD in the top-left corner shows live counts and distances.

---

## 📦 Module Breakdown

### `src/train.py`
Downloads the dataset from Roboflow and trains the YOLOv11m model. Run this once before inference.

### `src/detector.py`
The core library. Contains:
- `load_model(weights_path)` — loads a YOLO model from a `.pt` file
- `process_frame(model, frame)` — runs detection and returns an annotated frame
- `save_video_detection(model, video_path, output_name)` — processes a full video

### `src/run.py`
Thin entry point that batch-processes all `.mp4` files in a directory using `detector.py`.

---

## 🔑 Configuration

| Variable | Location | Description |
|---|---|---|
| `api_key` | `train.py` | Your Roboflow API key |
| `WEIGHTS` | `run.py` | Path to trained `.pt` weights |
| `CLIPS_DIR` | `run.py` | Directory containing input videos |
| `conf` | `run.py` / `detector.py` | Detection confidence threshold (default `0.5`) |

---

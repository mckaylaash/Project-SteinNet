# SteinNet: Visual Regression & Imitation Learning for Autonomous Game Agents

An end-to-end imitation learning pipeline and real-time vision-based agent deployed in **Steinworld**, a 2D browser-based tile MMORPG.

SteinNet bypasses traditional fragile heuristics (e.g., direct RAM/memory reading or hard-coded pixel-color scraping) by training convolutional neural networks directly on raw gameplay frames to map visual inputs to continuous mouse coordinates and discrete directional navigation.

---

## Architecture & System Overview

The system consists of two specialized networks sharing a common fine-tuned **ResNet-18** backbone (pretrained on ImageNet-1K):

1. **Spatial Regression Network (Lumbering / Gathering)**
   - **Head:** Linear regression layer outputting normalized 2D screen coordinates $(x, y) \in [0, 1]$.
   - **Objective:** Mean Squared Error (MSE) loss to penalize large Euclidean distance errors.
   - **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$, batch size 32) trained over 10 epochs.

2. **Directional Classification Network (Navigation)**
   - **Head:** 4-way classification layer predicting agent movement (`Forward`, `Left`, `Backward`, `Right`) along paths.
   - **Objective:** Cross-Entropy Loss.
   - **Optimization:** Adam optimizer over balanced frame batches.

3. **Hybrid State Machine Agent**
   - Implements a dual-state loop (`NAVIGATE` vs. `CHOP`).
   - Tracks path trajectories until resource targets are identified within interaction range, then switches dynamically to interaction mode.
   - Operates fully asynchronously via screen grabbing (`mss`/`PIL`), MPS device acceleration, and OS-level virtual inputs (`pyautogui`, `pynput`).

---

## Dataset & Data Collection

Training data was curated using a custom **Human-in-the-Loop** screen-capture and input-sync tool, collecting **1,621 unique labeled frames**:

* **1,115 Gathering / Lumbering Frames (68.8%):** Synchronized mouse-click coordinates mapped to screen dimensions upon player input.
* **506 Navigation Frames (31.2%):** Directional key-press inputs (`W/A/S/D`) synchronized with camera capture and path-texture labels.
* **Preprocessing:** Input frames resized to $224 \times 224$ and normalized using standard ImageNet mean and variance. Coordinate regression targets normalized to $[0, 1]$ relative to resolution.

---

## Experimental Results & Benchmarks

### 1. Directional Navigation Classification

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Forward** | 0.962 | 0.862 | **0.909** |
| **Left** | 0.933 | 0.875 | **0.903** |
| **Backward** | 0.952 | 1.000 | **0.976** |
| **Right** | 0.923 | 1.000 | **0.960** |

### 2. Spatial Coordinate Regression

* **Mean Squared Error (Normalized):** `0.00347`
* **RMSE per Axis:** `0.0833`
* **Median Pixel Error:** `54.9 px`
* **Hit Rate ($\le$ 50px radius):** `45.7%`
* **Hit Rate ($\le$ 80px radius):** `83.9%`

### 3. Latency & Inference Throughput

Benchmarked on Apple Silicon (macOS `mps` device):

| Model | Mean Latency | p95 Latency | Theoretical Max Throughput |
| :--- | :---: | :---: | :---: |
| **Click Regression** | 3.59 ms | 4.12 ms | **278.7 FPS** |
| **Navigation Classifier** | 3.67 ms | 4.02 ms | **272.8 FPS** |

*The agent's real-time execution loop runs with a target frame budget of 83.3 ms (12.0 FPS), well within model inference thresholds.*

---

## Project Structure

```text
Project-SteinNet/
├── data_collection.py          # Human-in-the-loop recording & labeling harness
├── preprocess.py               # Image transformation, resizing (224x224), label normalization
├── train.py                    # PyTorch training loops for regression and classification
├── agent.py                    # Real-time state machine inference loop and OS automation
├── performance_analysis.ipynb  # Evaluation scripts, confusion matrices, and latency profiling
├── weights/                    # Saved checkpoint weights (`stein_net_best.pth`)
└── requirements.txt            # Project dependencies

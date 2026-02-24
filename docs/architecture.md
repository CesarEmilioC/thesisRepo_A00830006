# System Architecture

This document describes the complete pipeline of the Paddle Tennis Movement Feedback System, from raw video input to movement quality prediction.

---

## Pipeline Overview

```
                        +-----------------+
                        |  Video Recording |
                        |   (Pre-recorded) |
                        +--------+--------+
                                 |
                                 v
                     +-----------------------+
                     |  Video Clip Cutting    |
                     |  (createClips.py)      |
                     |  OneDrive / External   |
                     +-----------+-----------+
                                 |
                     player{N}_part{M}_clip{K}_grade{G}.mp4
                                 |
                                 v
+-------------------------------------------------------------+
|                    POSE ESTIMATION                            |
|  module_poseEstimation.py -> analyze_video()                 |
|                                                              |
|  1. Load video frame-by-frame                                |
|  2. Run OpenPose inference (mobilenet_thin)                  |
|  3. Select target human (rightmost pelvis heuristic)         |
|  4. Extract pelvis, left shoulder, right elbow, right wrist  |
|  5. Record NaN for keypoints not detected (pelvis required)  |
|  6. Invert Y-axis to mathematical coordinate system          |
|  7. Interpolate missing keypoints (linear interpolation)     |
|  8. Compute pelvis-relative coordinates                      |
|  9. Save structured JSON with metadata + coordinates         |
+----------------------------+--------------------------------+
                             |
                  player{N}_part{M}_clip{K}_grade{G}.json
                  (saved to Coordinates/ directory)
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +------------+  +-------------+  +--------------+
     | Visualization| | LSTM Training| | Data Analysis |
     | module_     |  | module_LSTM |  | analyzeJSON  |
     | grapher.py  |  | .py         |  |              |
     +------------+  +------+------+  +--------------+
                            |
                            v
+-------------------------------------------------------------+
|                    LSTM TRAINING                             |
|  module_LSTM.py -> train_model()                             |
|                                                              |
|  1. Load all JSON coordinate files recursively               |
|  2. Concatenate pelvis + shoulder + elbow + wrist -> (N, 8)  |
|  3. Per-clip normalization (zero-mean, unit-variance)        |
|  4. Map grades (1-10) to 5 classes (0-4)                     |
|  5. Pad/truncate sequences to 90 frames                      |
|  6. Stratified 80/20 train/test split                        |
|  7. Compute balanced class weights                           |
|  8. Build Bidirectional LSTM model:                          |
|     BatchNorm -> BiLSTM(128) -> Dropout(0.30)                |
|     -> BiLSTM(64) -> Dropout(0.25) -> Dense(64) -> BatchNorm|
|     -> Dense(5, softmax)                                     |
|  9. Train with Adam optimizer, early stopping (patience=12)  |
| 10. Save model, training history, and evaluation metrics     |
+----------------------------+--------------------------------+
                             |
                     lstm_model.h5
                             |
                             v
+-------------------------------------------------------------+
|                    PREDICTION / INFERENCE                    |
|  module_LSTM.py -> predict_clip()                            |
|                                                              |
|  1. Load single JSON coordinate file                         |
|  2. Concatenate pelvis + shoulder + elbow + wrist features   |
|  3. Normalize sequence (same as training)                    |
|  4. Pad/truncate to 90 frames                                |
|  5. Feed through trained LSTM model                          |
|  6. Output: predicted class (0-4) with label                 |
+-------------------------------------------------------------+
```

---

## Module Responsibilities

| Module                     | Purpose                                           | CLI Command(s)          |
|----------------------------|---------------------------------------------------|-------------------------|
| `main.py`                  | CLI entry point, argument parsing, command routing | All                     |
| `module_poseEstimation.py` | OpenPose pose estimation, interpolation, and JSON generation | `pose`, `analyzeJSON` |
| `module_LSTM.py`           | LSTM model training, evaluation, and prediction    | `trainLSTM`, `predictLSTM`, `countGrades` |
| `module_grapher.py`        | Coordinate visualization and animation             | `plot`, `animate`       |
| `config.py`                | Central configuration (constants, paths, hyperparameters) | N/A              |

---

## Data Flow

1. **Video Recording**: Players are recorded performing bandeja shots. Videos are stored in OneDrive (not tracked by Git).

2. **Clip Cutting**: Full videos are cut into individual clips using `createClips.py` (external, in OneDrive). Each clip is named following the convention `player{N}_part{M}_clip{K}_grade{G}.mp4` where the grade is manually assigned based on technique quality.

3. **Pose Estimation**: The `pose` command processes each video clip through OpenPose, extracting pelvis, left shoulder, right elbow, and right wrist coordinates frame-by-frame. If a keypoint (other than the pelvis) is not detected in a frame, NaN is recorded and later filled via linear interpolation. Results are saved as JSON files in the `Coordinates/` directory.

4. **Training**: The `trainLSTM` command loads all coordinate JSONs, preprocesses sequences (normalization + padding to 90 frames), maps grades (1-10) to 5 classes, computes balanced class weights, and trains a Bidirectional LSTM model. Each frame produces 8 features: pelvis(2) + shoulder_relative(2) + elbow_relative(2) + wrist_relative(2). Results (model, metrics, plots) are saved to a timestamped folder in `Results/`.

5. **Prediction**: The `predictLSTM` command loads a trained model and a single JSON file, processes the coordinates (same normalization + padding as training), and outputs a predicted quality class (0-4: Very Low, Low, Medium, High, Excellent).

6. **Visualization**: The `plot` and `animate` commands generate static plots and animated visualizations of movement patterns from coordinate data. Supports both 3-keypoint (legacy) and 4-keypoint (with shoulder) JSON formats.

---

## Directory Structure

```
thesisRepo_A00830006/
|-- Coordinates/           <- Generated JSON coordinate files (by player/part)
|-- Samples/               <- Sample clips and JSONs for testing
|-- Source/
|   |-- main.py            <- CLI entry point
|   |-- config.py          <- Central configuration
|   |-- Models/            <- Trained LSTM models
|   |-- Modules/
|   |   |-- module_poseEstimation.py
|   |   |-- module_LSTM.py
|   |   +-- module_grapher.py
|   |-- openPoseRequirements/  <- OpenPose TensorFlow implementation
|   +-- Results/           <- Training experiment outputs
|-- docs/                  <- Project documentation
|-- environment.yml        <- Conda environment specification
|-- requirements.txt       <- Pip dependencies
|-- LICENSE                <- MIT License
+-- README.md              <- Main project documentation
```

---

## How to Run Each Part of the Pipeline

### Step 1: Pose Estimation (extract coordinates from video)
```bash
cd Source
python main.py pose --video "../Samples/clipSamples/player10_part1_clip0_grade7.mp4"
```
Or for an entire directory of videos:
```bash
python main.py pose --directory "../Videos/Clips/player10/part1"
```

### Step 2: Verify Data Quality
```bash
python main.py analyzeJSON --directory "../Coordinates"
```

### Step 3: Inspect Data Visually
```bash
python main.py plot --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --type all
python main.py animate --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
```

### Step 4: Check Grade Distribution
```bash
python main.py countGrades --directory "../Coordinates"
```

### Step 5: Train the LSTM Model
```bash
python main.py trainLSTM --directory "../Coordinates" --run_name Test01 --model_path "Models/lstm_model.h5"
```

### Step 6: Predict a Clip's Quality Class
```bash
python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --model_path "Models/lstm_model.h5"
```

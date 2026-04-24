# System Architecture

This document describes the complete pipeline of the Paddle Tennis Movement Feedback System, from raw video input to movement quality prediction. Three temporal architectures are implemented and compared: a Bidirectional **LSTM** (recommended model), a Bidirectional **GRU**, and a **TCN** (Temporal Convolutional Network).

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
                     |  Video Clip Cutting   |
                     |  (createClips.py)     |
                     |  OneDrive / External  |
                     +-----------+-----------+
                                 |
                     player{N}_part{M}_clip{K}_grade{G}.mp4
                                 |
                                 v
+-------------------------------------------------------------+
|                    POSE ESTIMATION                          |
|  module_poseEstimation.py -> analyze_video()                |
|                                                             |
|  1. Load video frame-by-frame                               |
|  2. Run OpenPose inference (mobilenet_thin)                 |
|  3. Select target human (rightmost pelvis heuristic)        |
|  4. Extract pelvis, left shoulder, right elbow, right wrist |
|  5. Record NaN for keypoints not detected (pelvis required) |
|  6. Invert Y-axis to mathematical coordinate system         |
|  7. Linear interpolation of missing non-pelvis keypoints    |
|  8. Compute pelvis-relative coordinates                     |
|  9. Save structured JSON with metadata + coordinates        |
+----------------------------+--------------------------------+
                             |
                  player{N}_part{M}_clip{K}_grade{G}.json
                  (saved to Coordinates/ directory)
                             |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
 +-------------+    +-----------------+    +-------------+
 | Visualization |   | Model Training  |    | Data Analysis|
 | module_       |   | module_LSTM.py  |    | analyzeJSON  |
 | grapher.py    |   | module_GRU.py   |    | countGrades  |
 |               |   | module_TCN.py   |    |              |
 +-------------+    +--------+--------+    +-------------+
                             |
                             v
+-------------------------------------------------------------+
|          MODEL TRAINING (LSTM / GRU / TCN)                  |
|  Shared data pipeline in module_data.py                     |
|                                                             |
|  1. Load all JSON coordinate files recursively              |
|  2. Concatenate pelvis + shoulder + elbow + wrist -> (N, 8) |
|  3. Per-clip normalization (zero-mean, unit-variance)       |
|  4. Map grades (1-10) to 5 classes (0-4)                    |
|  5. Pad/truncate sequences to 90 frames                     |
|  6. Stratified 80/20 train/test split                       |
|  7. Compute balanced class weights                          |
|  8. Build temporal model (see architectures below)          |
|  9. Gaussian-noise data augmentation (factor 2, sigma=0.02) |
| 10. Train with Adam + L2 regularization + ReduceLROnPlateau |
| 11. Early stopping (patience=15) on validation loss         |
| 12. Save model, training history, and evaluation metrics    |
+----------------------------+--------------------------------+
                             |
             {model}Model_TestXX_DD-MM-YYYY.h5  (auto-named)
                             |
                             v
+-------------------------------------------------------------+
|                    PREDICTION / INFERENCE                   |
|  module_{LSTM,GRU,TCN}.py -> predict_clip()                 |
|                                                             |
|  1. Auto-detect the most recent saved model                 |
|  2. Load single JSON coordinate file                        |
|  3. Concatenate pelvis + shoulder + elbow + wrist features  |
|  4. Normalize sequence (same as training)                   |
|  5. Pad/truncate to 90 frames                               |
|  6. Feed through the selected model                         |
|  7. Output: predicted class (0-4) with label                |
+-------------------------------------------------------------+
```

---

## Model Architectures

### Bidirectional LSTM (recommended)
```
BatchNormalization
  -> BiLSTM(64, L2=1e-4) -> Dropout(0.40)
  -> BiLSTM(32, L2=1e-4) -> Dropout(0.35)
  -> Dense(32, relu, L2=1e-4) -> BatchNormalization
  -> Dense(5, softmax)
```

### Bidirectional GRU (baseline)
```
BatchNormalization
  -> BiGRU(64, L2=1e-4) -> Dropout(0.40)
  -> BiGRU(32, L2=1e-4) -> Dropout(0.35)
  -> Dense(32, relu, L2=1e-4) -> BatchNormalization
  -> Dense(5, softmax)
```

### Temporal Convolutional Network (non-recurrent baseline)
```
BatchNormalization
  -> Conv1D(64, kernel=3, dilation=1, causal) -> BN -> Dropout(0.40)
  -> Conv1D(64, kernel=3, dilation=2, causal) -> BN -> Dropout(0.40)
  -> Conv1D(64, kernel=3, dilation=4, causal) -> BN -> Dropout(0.40)
  -> GlobalAveragePooling1D
  -> Dense(32, relu) -> Dropout(0.40) -> Dense(5, softmax)
```

All three models share the same data pipeline, augmentation, class weights, optimizer (Adam), loss (sparse categorical cross-entropy), and learning-rate schedule.

---

## Module Responsibilities

| Module                     | Purpose                                                           | CLI Command(s)                |
|----------------------------|-------------------------------------------------------------------|-------------------------------|
| `main.py`                  | CLI entry point, argument parsing (argparse), command routing     | All                           |
| `config.py`                | Central configuration (constants, paths, hyperparameters)         | N/A                           |
| `module_poseEstimation.py` | OpenPose inference, interpolation, and JSON generation            | `pose`, `analyzeJSON`         |
| `module_data.py`           | Shared data loading, preprocessing, evaluation, and analysis      | (used internally)             |
| `module_LSTM.py`           | Bidirectional LSTM definition, training, prediction               | `trainLSTM`, `predictLSTM`    |
| `module_GRU.py`            | Bidirectional GRU definition, training, prediction                | `trainGRU`, `predictGRU`      |
| `module_TCN.py`            | TCN definition (dilated causal convolutions), training, prediction| `trainTCN`, `predictTCN`      |
| `module_grapher.py`        | Coordinate visualization, animation, and thesis-figure generation | `plot`, `animate`, thesis cmds |

---

## Data Flow

1. **Video Recording**: Players are recorded performing *bandeja* shots. Videos are stored in OneDrive (not tracked by Git).

2. **Clip Cutting**: Full videos are cut into individual clips using `createClips.py` (external, in OneDrive). Each clip is named following the convention `player{N}_part{M}_clip{K}_grade{G}.mp4` where the grade is manually assigned based on technique quality.

3. **Pose Estimation**: The `pose` command processes each video clip through OpenPose, extracting pelvis, left shoulder, right elbow, and right wrist coordinates frame-by-frame. If a non-pelvis keypoint is not detected in a frame, NaN is recorded and later filled via linear interpolation. Frames where the pelvis is not detected are discarded. Results are saved as JSON files in the `Coordinates/` directory. The optional `--save-frames-mosaic` flag additionally produces a PNG mosaic with up to 5 evenly-spaced frames and a color legend per joint.

4. **Training**: The `trainLSTM` / `trainGRU` / `trainTCN` commands load all coordinate JSONs, preprocess sequences (normalization + padding to 90 frames), map grades (1–10) to 5 classes, compute balanced class weights, apply Gaussian-noise data augmentation (factor 2), and train the corresponding temporal model. Each frame produces 8 features: pelvis(2) + shoulder_relative(2) + elbow_relative(2) + wrist_relative(2). The test number and model filename are auto-detected from the contents of the `Results/` and `Models/` folders. Outputs (model, metrics, plots) are saved to a timestamped folder in `Results/`.

5. **Prediction**: The `predictLSTM` / `predictGRU` / `predictTCN` commands auto-detect the most recent trained model of the corresponding architecture, load a single JSON file, process the coordinates (same normalization + padding as training), and output a predicted quality class (0–4: Very Low, Low, Medium, High, Excellent).

6. **Visualization**: The `plot` and `animate` commands generate static plots and animated GIFs of movement patterns from coordinate data. Supports both 3-keypoint (legacy) and 4-keypoint (current, with shoulder) JSON formats.

7. **Thesis artifacts**: A set of CLI commands (`regenPlots`, `labelDist`, `spearman`, `sysOutput`, `datasetStats`, `thesisMosaic`) regenerate publication-ready figures and statistics from saved training artifacts, without retraining.

---

## Directory Structure

```
thesisRepo_A00830006/
|-- Coordinates/           <- Generated JSON coordinate files (by player/part)
|-- Demo/                  <- Google Colab demo notebook and sample files
|-- Samples/               <- Sample clips and JSONs for testing
|-- Source/
|   |-- main.py            <- CLI entry point (argparse-based)
|   |-- config.py          <- Central configuration
|   |-- Models/            <- Trained models (auto-named: {model}Model_TestXX_DD-MM-YYYY.h5)
|   |-- Modules/
|   |   |-- module_poseEstimation.py
|   |   |-- module_data.py
|   |   |-- module_LSTM.py
|   |   |-- module_GRU.py
|   |   |-- module_TCN.py
|   |   +-- module_grapher.py
|   |-- openPoseRequirements/  <- OpenPose TensorFlow implementation
|   +-- Results/           <- Training experiment outputs (per run)
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
Add `--save-frames-mosaic` to also generate a 5-frame PNG mosaic with keypoints overlaid.

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

### Step 5: Train a Model (choose one)
```bash
python main.py trainLSTM --directory "../Coordinates"
python main.py trainGRU  --directory "../Coordinates"
python main.py trainTCN  --directory "../Coordinates"
```
The test number and model filename are auto-detected (e.g., if `LSTM_Test03` exists, the next run creates `LSTM_Test04`).

### Step 6: Predict a Clip's Quality Class
```bash
python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
python main.py predictGRU  --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
python main.py predictTCN  --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
```
The most recent saved model of the corresponding architecture is loaded automatically.

### Step 7 (Optional): Regenerate thesis artifacts
```bash
python main.py regenPlots --directory "../Coordinates"   # learning curves, CM, class distribution
python main.py labelDist  --directory "../Coordinates"   # dataset-wide class histogram
python main.py spearman   --directory "../Coordinates"   # Spearman rho for the 3 final models
python main.py sysOutput  --directory "../Coordinates"   # system output comparison figure
python main.py datasetStats --directory "../Coordinates" # clips per player, class, and grade
```

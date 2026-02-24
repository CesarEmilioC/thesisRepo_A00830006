# Paddle Tennis Movement Feedback System

This project addresses the need for accessible, data-driven feedback for **amateur paddle tennis** players. It builds on research in **motion tracking** and **action recognition** to enhance sports performance through a machine learning-based system that analyzes video-captured body movements and provides actionable insights.

At the core of the system is a **pose estimation module** that tracks key movements -- such as the _bandeja_ shot. Video data from both professional and amateur players was collected under controlled conditions, and each frame was annotated to map body joint positions. These mappings were categorized based on the quality of the technique and used to train a supervised recursive model. This model evaluates new player movements by comparing them to ideal movement patterns observed in professionals.

---

## Reproducing the Development Environment

To replicate the development environment on another machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/CesarEmilioC/thesisRepo_A00830006.git
   cd THESISREPO_A00830006
   ```

2. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

   The system requires **Python 3.11+** and **TensorFlow 2.18+** (latest stable).

This ensures that all dependencies and versions used in development are reproduced accurately.

---

## Project Structure

```bash
THESISREPO_A00830006/
|
|-- Coordinates/                    # JSON coordinate files organized by player/part
|   |-- player3/
|   |-- player4/
|   |-- player5/
|   |-- player6/
|   |-- player9/
|   +-- player10/
|
|-- Samples/                        # Sample data for testing and demonstration
|   |-- clipSamples/                # Sample video clips (.mp4)
|   +-- coordinateSamples/          # Sample coordinate files (.json)
|
|-- Source/
|   |-- main.py                     # CLI entry point
|   |-- config.py                   # Central configuration (constants, paths, hyperparameters)
|   |-- Models/
|   |   +-- lstm_model.h5           # Trained LSTM model
|   |-- Modules/
|   |   |-- module_poseEstimation.py  # OpenPose pose estimation + interpolation
|   |   |-- module_LSTM.py            # LSTM model training, evaluation, prediction
|   |   +-- module_grapher.py         # Coordinate visualization and animation
|   |-- openPoseRequirements/       # TensorFlow-based OpenPose implementation
|   +-- Results/                    # Training experiment outputs (timestamped)
|
|-- docs/                           # Project documentation
|   |-- data_dictionary.md          # JSON format and field descriptions
|   +-- architecture.md             # System pipeline and module responsibilities
|
|-- environment.yml
|-- requirements.txt
|-- LICENSE
+-- README.md
```

### Folder Descriptions

- **Coordinates/** -- Contains JSON coordinate files generated from OpenPose, organized by player and video part. Each file stores pelvis, left shoulder, right elbow, and right wrist coordinates with metadata.
- **Samples/** -- Stores sample videos and their corresponding JSONs for testing and visualization.
  - `clipSamples/`: Short video clips of _bandeja_ shots.
  - `coordinateSamples/`: Corresponding coordinate data for each clip.
- **Source/config.py** -- Central configuration file containing all constants, file paths, OpenPose keypoint indices, LSTM hyperparameters, and visualization settings. All magic numbers are defined here.
- **Source/Modules/** -- Contains Python modules for pose estimation, LSTM model training/testing, and data visualization.
- **Source/Models/** -- Stores the trained LSTM model (`lstm_model.h5`).
- **Source/main.py** -- CLI entry point that manages pose extraction, analysis, and LSTM operations.
- **docs/** -- Detailed project documentation including data dictionary and architecture overview.

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[Data Dictionary](docs/data_dictionary.md)** -- Complete description of the JSON coordinate file format, metadata fields, coordinate system, grade scale, interpolation method, and OpenPose keypoint indices.
- **[Architecture](docs/architecture.md)** -- System pipeline overview, module responsibilities, data flow, and step-by-step guide for running each part of the code.

---

## Video Dataset (Independent from GitHub Repository)

The **Videos** folder is hosted independently on OneDrive (not tracked in Git).
It contains the raw recordings, pre-cut clips organized by player and part, and the JSON timestamp files used to generate clips.

Dataset link:
https://tecmx-my.sharepoint.com/:f:/g/personal/a00830006_tec_mx/EuvOsh32lh5El-Aitld6c9UBhsb97xw9q9HbERRJAxOjwg?e=3RdkXB

### Video Folder Structure

```bash
Videos/
|-- Original Videos/           # Full uncut recordings per player
|   |-- player1/
|   |   |-- player1_part1.mp4
|   |   +-- player1_part2.mp4
|   +-- ...
|
|-- Clips/                     # Pre-cut individual bandeja shots
|   |-- player1/
|   |   |-- part1/
|   |   |   |-- player1_part1_clip1_gradeY.mp4
|   |   |   +-- ...
|   |   +-- part2/
|   |       +-- ...
|   +-- ...
|
|-- Original Video Cuts/       # JSON files with timestamp cuts per video
|   |-- player1/
|   |   |-- player1_part1.json
|   |   +-- ...
|   +-- ...
|
|-- createClips.py             # Script to cut original videos into clips
+-- playerSamples_trainingData.xls  # Grade labels spreadsheet
```

---

## Running the Code

The repository provides a command-line tool (`main.py`) with multiple modules to perform all key operations: pose estimation, visualization, animation, training, prediction, and data validation.

### General Command Structure

All commands follow the same syntax:

```bash
cd Source
python main.py <command> [arguments]
```

Run without arguments to display available options:

```bash
python main.py --help
```

---

### 1. Pose Estimation with OpenPose

Extract body joint coordinates from pre-recorded video files. The system extracts **4 keypoints**: pelvis, left shoulder, right elbow, and right wrist using the Body-25 OpenPose model.

**From a single video:**

```bash
cd Source
python main.py pose --video "../Samples/clipSamples/player10_part1_clip0_grade7.mp4"
```

**From an entire folder of videos:**

```bash
cd Source
python main.py pose --directory "../Videos/Clips/player10/part1"
```

Arguments:
- `--video`: Path to a single video file (.mp4).
- `--directory`: Path to a folder containing multiple .mp4 files.
- `--model`: OpenPose model type (`mobilenet_thin`, `cmu`, etc.). Default: `mobilenet_thin`.
- `--show_video`: Display the video window during processing.
- `--resize`: Input resolution for OpenPose (e.g., `'432x368'`). Default: native resolution.
- `--resize-out-ratio`: Upsample ratio for OpenPose inference. Default: `4.0`.

Output: JSON coordinate files saved to `Coordinates/player{N}/part{M}/` (subdirectories are created automatically).

#### Skip Existing Files

When processing a directory, the system checks if a JSON coordinate file already exists for each clip. If found, the clip is skipped with a `[SKIP]` message, avoiding redundant processing. This means you can safely re-run pose estimation on a directory to process only the missing clips (e.g., after an interruption).

#### Memory Optimization

When processing a directory of videos, the OpenPose model is initialized **once** and reused for all clips. This prevents memory accumulation (OOM errors) that would otherwise occur when re-initializing TensorFlow for each clip. If an individual clip fails, processing continues with the remaining clips and a summary is printed at the end.

#### Interpolation of Missing Keypoints

When OpenPose fails to detect a keypoint in a frame (e.g., the shoulder is occluded), the system records `NaN` for that keypoint and continues processing. After all frames are processed, missing values are filled using **linear interpolation** between the nearest valid detections.

Requirements for interpolation:
- The **pelvis must be detected** in the frame (pelvis is the reference point; frames without pelvis are discarded entirely).
- At least **2 valid detections** of the missing keypoint must exist across the clip for interpolation to work.

This approach produces more complete and continuous coordinate sequences compared to discarding frames with any missing keypoint. The number of interpolated keypoint-frames is recorded in the JSON metadata field `interpolated_frames`.

---

### 2. Plot Coordinates (Movement Visualization)

Generates trajectory and temporal plots from JSON coordinate data. Supports both 3-keypoint (legacy) and 4-keypoint (with shoulder) JSON formats.

```bash
cd Source
python main.py plot --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --type all
```

Plot types available:
- `original`: Absolute pixel coordinates (pelvis + wrist + elbow + shoulder trajectories).
- `relative`: Coordinates relative to pelvis (isolates arm movement).
- `temporal`: Joint position evolution over time (X and Y separately).
- `3d`: 3D spatial trajectory (X, Y, Time).
- `all`: Generates all plot types.

---

### 3. Animate Movement

Creates an animation of the detected movement sequence showing pelvis, shoulder, elbow, and wrist motion. When shoulder data is available, the animation draws the complete kinematic chain: Pelvis -> Shoulder -> Elbow -> Wrist.

```bash
cd Source
python main.py animate --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
```

---

### 4. Train LSTM Model

Trains the Bidirectional LSTM network using all player coordinate data.

```bash
cd Source
python main.py trainLSTM --directory "../Coordinates" --run_name Test01 --model_path "Models/lstm_model.h5"
```

Arguments:
- `--directory` (required): Path to directory containing JSON coordinate files.
- `--run_name` (required): Name for the results folder (inside `Results/`).
- `--model_path`: Path to save the trained model. Default: `Models/lstm_model.h5`.

#### LSTM Training Results

Each training run creates a timestamped folder inside `Results/` containing:

```bash
Results/
+-- Test01_23-02-26/
    |-- training_history.json       # Loss and accuracy values for each epoch
    |-- learning_curves.png         # Training/validation loss and accuracy plots
    |-- confusion_matrix.png        # Confusion matrix on the test set
    |-- class_distribution.png      # True vs predicted class histogram
    |-- classification_report.txt   # Precision, recall, and F1-score summary
    +-- lstm_model.h5               # Copy of the trained model for reproducibility
```

#### LSTM Architecture

The model uses a Bidirectional LSTM architecture:
- Input: Sequences of 90 frames x 8 features (pelvis + shoulder_rel + elbow_rel + wrist_rel, 2D coordinates each)
- BatchNormalization -> BiLSTM(128) -> Dropout(0.30) -> BiLSTM(64) -> Dropout(0.25)
- Dense(64, relu) -> BatchNormalization -> Dense(5, softmax)
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy | Early stopping: patience=12
- Class weights: Balanced (computed automatically to handle class imbalance)

#### 5-Class Classification

Grades (1-10) are mapped to 5 quality classes for classification:

| Grade | Class | Label      |
|-------|-------|------------|
| 1-2   | 0     | Very Low   |
| 3-4   | 1     | Low        |
| 5-6   | 2     | Medium     |
| 7-8   | 3     | High       |
| 9-10  | 4     | Excellent  |

#### Normalization

Each clip is independently normalized using per-feature zero-mean, unit-variance normalization. The same normalization is applied during both training and inference via the shared `normalize_sequence()` function.

#### Backward Compatibility

The LSTM module supports both 3-keypoint (6 features) and 4-keypoint (8 features) JSON formats. When loading older JSONs without shoulder data, a warning is printed and the sequence is loaded with 6 features. However, a model trained on 8-feature data cannot be used to predict 6-feature clips and vice versa.

---

### 5. Predict Clip Quality Class (LSTM Inference)

Predicts the quality class of a clip using the trained LSTM model.

```bash
cd Source
python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --model_path "Models/lstm_model.h5"
```

Output: Predicted class (0-4) with label (e.g., "High") printed to the console.

---

### 6. Count Clips per Grade

Counts how many clips exist per grade label across the dataset. Shows both the raw grade distribution (1-10) and the 5-class distribution.

```bash
cd Source
python main.py countGrades --directory "../Coordinates"
```

---

### 7. Analyze JSON Validity

Computes the proportion of valid frames (frames where the pelvis was detected) relative to total video frames.

```bash
cd Source
python main.py analyzeJSON --directory "../Coordinates"
```

Output: Percentage summary of valid frame data per JSON and the overall mean.

---

## Configuration

All configurable parameters are centralized in `Source/config.py`:

| Section | Examples |
|---------|----------|
| **Paths** | `COORDINATES_DIR`, `RESULTS_DIR`, `MODELS_DIR` |
| **OpenPose Keypoints** | `KEYPOINT_PELVIS=8`, `KEYPOINT_LEFT_SHOULDER=5`, `KEYPOINT_RIGHT_ELBOW=3`, `KEYPOINT_RIGHT_WRIST=4` |
| **JSON Field Names** | `FIELD_PELVIS`, `FIELD_SHOULDER_ORIGINAL`, `FIELD_ELBOW_ORIGINAL`, `FIELD_WRIST_RELATIVE`, etc. |
| **LSTM Hyperparameters** | `LSTM_UNITS_L1=128`, `DROPOUT_L1=0.30`, `MAX_SEQUENCE_LENGTH=90`, `NUM_FEATURES=8`, `NUM_CLASSES=5`, `EPOCHS=80` |
| **Classification** | `CLASS_LABELS`, `grade_to_class()` |
| **Visualization** | `ANIMATION_PLAYBACK_SPEED=0.2`, `COLORMAP_RANGE_START=0.3` |

To modify any parameter (e.g., change the LSTM architecture or max sequence length), edit `config.py` instead of searching through module code.

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tf_slim'` | Missing TensorFlow Slim dependency | `pip install tf-slim` |
| `[ERROR] Video file not found: ...` | Invalid video path | Check the path is correct and the file exists |
| `[ERROR] No valid sequences loaded` | Empty or corrupted JSON files in directory | Run `analyzeJSON` to check data quality |
| `[ERROR] Model file not found: ...` | No trained model at the specified path | Train a model first with `trainLSTM` |
| `[WARNING] Skipping empty sequence: ...` | JSON file has no valid frames | Re-run pose estimation or remove the file |
| `[WARNING] No shoulder data in ...` | Older 3-keypoint JSON format | Re-run pose estimation to generate 4-keypoint data |
| `[WARNING] No pelvis detections in ...` | OpenPose failed to detect pelvis in all frames | Check video quality or camera angle |

---

## Authors

**Cesar Emilio Castano Marin**
Thesis Student
Computer Science Master's -- Tecnologico de Monterrey

**Marcial Roberto Leyva Fernandez**
Thesis Advisor
School of Engineering and Sciences -- Tecnologico de Monterrey

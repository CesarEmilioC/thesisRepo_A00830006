# Paddle Tennis Movement Feedback System

This project addresses the need for accessible, data-driven feedback for **amateur paddle tennis** players. It builds on research in **motion tracking** and **action recognition** to enhance sports performance through a machine learning-based system that analyzes video-captured body movements and provides real-time, actionable insights.

At the core of the system is a **pose estimation module** that tracks key movements—such as the _bandeja_ shot. Video data from both professional and amateur players was collected under controlled conditions, and each frame was annotated to map body joint positions. These mappings were categorized based on the quality of the technique and used to train a supervised recursive model. This model evaluates new player movements by comparing them to ideal movement patterns observed in professionals.

A user-friendly digital interface delivers instant feedback and highlights technique deviations, allowing players to make real-time adjustments and foster structured skill development. This system contributes to the fields of **AI**, **computer vision**, and **sports analytics**, providing a practical tool for improving paddle tennis performance.

---

## 🛠️ Reproducing the Development Environment

To replicate the development environment on another machine:

1. Clone the repository:  
   ```bash
   git clone https://github.com/CesarEmilioC/thesisRepo_A00830006.git
   cd THESISREPO_A00830006
   ```

2. Create the Conda environment from the YAML file:  
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:  
   ```bash
   conda activate tf1
   ```

4. (Optional) Install additional pip packages:  
   ```bash
   pip install -r requirements.txt
   ```

This ensures that all dependencies and versions used in development are reproduced accurately.

---

## 📁 Project Structure (GitHub Repository)

```bash
THESISREPO_A00830006/
│
├── Coordinates/
│   ├── player1/
│   ├── player2/
|   ├── ...
│   ├── player9/
│   └── player10/
│
├── Samples/
│   ├── clipSamples/
│   │   ├── player10_part1_clip0_grade7.mp4
│   │   └── player10_part1_clip3_grade8.mp4
│   │
│   └── coordinateSamples/
│       ├── player10_part1_clip0_grade7.json
│       └── player10_part1_clip3_grade8.json
│
├── Source/
│   ├── Models/
│   │   └── lstm_model.h5
│   │
│   ├── Modules/
│   │   ├── module_grapher.py
│   │   ├── module_LSTM.py
│   │   └── module_poseEstimation.py
│   │
│   ├── openPoseRequirements/
│   └── main.py
│
├── environment.yml
├── LICENSE
├── README.md
└── requirements.txt
```

### Folder Descriptions

- **Coordinates/** – Contains JSON coordinate files generated from OpenPose, organized by player and video part.
- **Samples/** – Stores sample videos and their corresponding JSONs for testing and visualization.  
  - `clipSamples/`: Short video clips of _bandeja_ shots.  
  - `coordinateSamples/`: Corresponding coordinate data for each clip.  
- **Source/Modules/** – Contains Python modules for pose estimation, LSTM model training/testing, and data visualization.
- **Source/Models/** – Stores the trained LSTM model (`lstm_model.h5`).
- **Source/main.py** – CLI entry point that manages pose extraction, analysis, and LSTM operations.
- **environment.yml / requirements.txt** – Environment and dependency definitions.
- **README.md / LICENSE** – Documentation and licensing information.

---

## 🎥 Video Dataset (Independent from GitHub Repository)

The **Videos** folder is hosted independently on OneDrive (not tracked in Git).  
It contains the raw recordings, pre-cut clips organized by player and part, and the JSON timestamp files used to generate clips.

🔗 Dataset link:  
https://tecmx-my.sharepoint.com/:f:/g/personal/a00830006_tec_mx/EuvOsh32lh5El-Aitld6c9UBhsb97xw9q9HbERRJAxOjwg?e=3RdkXB

### Folder Structure (with `part1`, `part2`, ... inside `Clips/`)

```bash
Videos/
│
├── Original Videos/
│   ├── player1/
│   │   ├── player1_part1.mp4
│   │   ├── player1_part2.mp4
│   │   ...
│   ├── player2/
│   │   ├── player2_part1.mp4
│   │   ...
│   ...
│   └── player10/
│       ├── player10_part1.mp4
│       ...
│
├── Clips/
│   ├── player1/
│   │   ├── part1/
│   │   │   ├── player1_part1_clip1_gradeY.mp4
│   │   │   ├── player1_part1_clip2_gradeX.mp4
│   │   │   ...
│   │   ├── part2/
│   │   │   ├── player1_part2_clip1_gradeY.mp4
│   │   │   ...
│   │   ...
│   └── player10/
│       ├── part1/
│       │   ├── player10_part1_clip1_gradeY.mp4
│       │   ...
│       ├── part2/
│       │   ├── player10_part2_clip1_gradeX.mp4
│       │   ...
│
├── Original Video Cuts/
│   ├── player1/
│   │   ├── player1_part1.json
│   │   ├── player1_part2.json
│   │   ...
│   ...
│   └── player10/
│       ├── player10_part1.json
│       ├── player10_part2.json
│       ├── player10_part3.json
│       ...
│
├── createClips.py
└── playerSamples_trainingData.xls
```

---

## 🚀 Running the Code

The repository provides a command-line tool (`main.py`) with multiple modules to perform all key operations: pose estimation, visualization, animation, training, prediction, and data validation.

### 🧩 General Command Structure

All commands follow the same syntax:

```bash
python main.py <command> [arguments]
```

Run without arguments to display available options:

```bash
python main.py --help
```

---

### 🎯 1. Pose Estimation with OpenPose

**From a single video:**

```bash
cd Source
python main.py pose --camera "../Samples/clipSamples/player10_part1_clip0_grade7.mp4"
```

**From an entire folder of videos:**

```bash
cd Source
python main.py pose --directory "../Videos/Clips/player10/part1"
```

Arguments:
- `--camera`: Path to a video file or camera index (`0` for webcam).
- `--directory`: Path to a folder containing multiple videos.
- `--model`: OpenPose model type (`mobilenet_thin`, `cmu`, etc.).
- `--show_video`: Display the video during processing.
- `--resize`: Input resolution (e.g., `'432x368'`).

Output: JSON coordinate files saved to the `Coordinates/` folder.

---

### 📊 2. Plot Coordinates (Movement Visualization)

Generates trajectory and temporal plots from JSON data.

```bash
cd Source
python main.py plot --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --type all
```

Types available:
- `original`: Raw coordinates.
- `relative`: Relative to the pelvis.
- `temporal`: Joint evolution over time.
- `3d`: 3D spatial trajectory.
- `all`: Generates all plot types.

---

### 🎞️ 3. Animate Movement

Creates an animation of a motion sequence using the extracted coordinates.

```bash
cd Source
python main.py animate --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
```

---

### 🧠 4. Train LSTM Model

Trains the LSTM network using all player coordinate data.

```bash
cd Source
python main.py trainLSTM --directory "../Coordinates" --model-path "Models/lstm_model.h5"
```

Output: A trained model saved as `lstm_model.h5` inside the `Source/Models/` directory.

#### 📂 LSTM Training Results

Every time you train an LSTM model using this command:

```bash
cd Source
python main.py trainLSTM --directory "../Coordinates" --run_name MyExperiment --model_path "Models/lstm_model.h5"
```

A new folder is automatically created inside the Results/ directory using the provided run_name. Each experiment folder stores all training outputs in a fully reproducible format, including:

```bash
Results/
└── MyExperiment/
    ├── training_history.json       ← Loss and accuracy values for each epoch
    ├── learning_curves.png         ← Training/validation loss and accuracy plots
    ├── confusion_matrix.png        ← Confusion matrix on the test set
    ├── class_distribution.png      ← True vs predicted class histogram
    ├── classification_report.txt   ← Precision, recall, and F1-score summary
    └── lstm_model.h5               ← The trained neural network model
```

---

### 🔮 5. Predict Clip Grade (LSTM Inference)

Predicts the quality grade of a clip using the trained LSTM model.

```bash
cd Source
python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --model-path "Models/lstm_model.h5"
```

Output: Printed predicted grade on the console.

---

### 📈 6. Count Clips per Grade

Counts how many clips exist per grade label across the dataset.

```bash
cd Source
python main.py countGrades --directory "../Coordinates"
```

---

### 🧩 7. Analyze JSON Validity

Computes the proportion of valid frames (frames where all joints were detected).

```bash
cd Source
python main.py analyzeJSON --directory "../Coordinates"
```

Output: Percentage summary of valid frame data per JSON and the overall mean.

---

## 👤 Authors

**Cesar Emilio Castaño Marin**  
Thesis Student
Computer Science Master’s – Tecnológico de Monterrey  

**Marcial Roberto Leyva Fernández**  
Thesis Advisor  
School of Engineering and Sciences – Tecnológico de Monterrey
# Paddle Tennis Movement Feedback System

This project addresses the need for accessible, data-driven feedback for **amateur paddle tennis** players. It builds on research in **motion tracking** and **action recognition** to enhance sports performance through a machine learning-based system that analyzes video-captured body movements and provides real-time, actionable insights.

At the core of the system is a **pose estimation module** that tracks key movements—such as the _bandeja_ shot. Video data from both professional and amateur players was collected under controlled conditions, and each frame was annotated to map body joint positions. These mappings were categorized based on the quality of the technique and used to train a supervised recursive model. This model evaluates new player movements by comparing them to ideal movement patterns observed in professionals.

A user-friendly digital interface delivers instant feedback and highlights technique deviations, allowing players to make real-time adjustments and foster structured skill development. This system contributes to the fields of **AI**, **computer vision**, and **sports analytics**, providing a practical tool for improving paddle tennis performance.

---

## 🛠️ Reproducing the Development Environment

To replicate the development environment on another machine:

1. Clone the repository:
   git clone https://github.com/CesarEmilioC/thesisRepo_A00830006.git  
   cd THESISREPO_A00830006

2. Create the Conda environment from the YAML file:
   conda env create -f environment.yml

3. Activate the environment:
   conda activate tf1

4. (Optional) Install additional pip packages:  
   If you are using `requirements.txt` for pip-only dependencies:  
   pip install -r requirements.txt

This ensures that all dependencies and versions used in development are reproduced accurately.

---

## 📁 Project Structure (GitHub Repository)

THESISREPO_A00830006/  
│  
├── Samples/  
│   ├── clipSamples/  
│   │   ├── player1_part1_clip0_grade7.json  
│   │   ├── player1_part1_clip1_grade6.json  
│   │   ...  
│   │   └── player10_partX_clipZ_gradeY.json  
│   │  
│   └── coordinateSamples/  
│       ├── player1_part1_clip0_grade7.json  
│       ├── player1_part1_clip1_grade6.json  
│       ...  
│       └── player10_partX_clipZ_gradeY.json  
│  
├── Source/  
│   ├── Modules/  
│   │   ├── __pycache__/  
│   │   ├── module_grapher.py  
│   │   ├── module_LSTMmodel.py  
│   │   ├── module_poseEstimation.py  
│   │   ├── module_train.py  
│   │   └── module_test.py  
│   ├── openPoseRequirements/  
│   └── main.py  
│  
├── environment.yml  
├── requirements.txt  
└── README.md  

- `Source/Modules/`: Contains all functional Python modules for data processing, model training/testing, pose estimation, and visualization.  
- `Source/main.py`: Main script to execute the system pipeline using command-line arguments.  
- `openPoseRequirements/`: Contains setup and dependencies related to OpenPose.  
- `Samples/coordinateSamples`: Stores output JSON files containing metadata and joint coordinates from processed clips.  
- `Samples/clipSamples`: Stores sample input test video clips for development and testing.  
- `environment.yml`: Conda environment specification.  
- `requirements.txt`: Additional pip-based dependencies.  
- `README.md`: Project documentation (you are here!).  

---

## 🎥 Video Dataset (Independent from GitHub Repository)

The **Videos** folder is hosted independently on OneDrive (not tracked in Git).  
It contains the raw recordings, pre-cut clips organized by player and part, and the JSON timestamp files used to generate clips.

### Folder Structure (updated with `part1`, `part2`, ... inside `Clips/`)

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
│   ├── player2/
│   │   ├── part1/
│   │   │   ├── player2_part1_clip1_gradeY.mp4
│   │   │   ...
│   │   ├── part2/
│   │   │   ├── player2_part2_clip1_gradeZ.mp4
│   │   │   ...
│   │   ...
│   ...
│   └── player10/
│       ├── part1/
│       │   ├── player10_part1_clip1_gradeY.mp4
│       │   │   ...
│       ├── part2/
│       │   ├── player10_part2_clip1_gradeX.mp4
│       │   │   ...
│       ...
│
├── Original Video Cuts/
│   ├── player1/
│   │   ├── player1_part1.json
│   │   ├── player1_part2.json
│   │   ...
│   ├── player2/
│   │   ├── player2_part1.json
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

### Naming Conventions

- Players: `player1` ... `player10`
- Original videos: `player{N}_part{M}.mp4`
- Cut definitions (JSON): `player{N}_part{M}.json`
- Clips: stored under `Clips/player{N}/part{M}/`
  - Filenames: `player{N}_part{M}_clip{K}_grade{G}.mp4`
    - `N`: player id (1–10)
    - `M`: part number (1, 2, …)
    - `K`: sequential clip index within that part (starting from 1)
    - `G`: grade assigned by a professional (e.g., 6, 7, 8, 9)

### Folder Descriptions

- `Original Videos`: Raw recorded videos of players performing _bandeja_ shots.  
- `Clips`: Extracted clips from the original videos, each annotated with a grade given by a professional padel player.  
- `Original Video Cuts`: JSON files containing metadata, timetstamps and grading for clips to be cut from the original videos.  
- `createClips.py`: Python function to automatically generate clips from original videos into their respective player folders.  
- `playerSamples_trainingData.xls`: Excel file with metadata related to all 10 players in the dataset.  

---

## 🚀 Running the Code

The `main.py` script uses a command-line interface via `argparse` to allow modular execution of the system components. You can run specific tasks such as pose extraction or coordinate plotting by providing the appropriate command-line flags.

### Example Commands

**To extract pose data from a test video using OpenPose:**  
cd Source  
python main.py pose --camera ..\Samples\clipSamples\<video_filename>.mp4  

This will process the input video and save the extracted joint coordinates (shoulder, elbow, wrist) to a JSON file inside the `Samples/coordinateSamples` directory.

---

**To visualize a coordinate file (previously extracted):**  
cd Source  
python main.py plot --file ..\Samples\coordinateSamples\<coordinate_filename>.json  

This generates a visual representation of the movement trajectory from the coordinate file. It is useful for comparing technical patterns and analyzing shot quality.

> More commands (e.g., training or testing the LSTM model) will be added as development progresses.  

---

## 👤 Authors

**Cesar Emilio Castaño Marin**  
Master’s Thesis Student – ITESM  

**Marcial Roberto Leyva Fernández**  
Thesis Advisor  
School of Engineering and Sciences – Tecnológico de Monterrey  

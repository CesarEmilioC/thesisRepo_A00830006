# Paddle Tennis Movement Feedback System

This project addresses the need for accessible, data-driven feedback for **amateur paddle tennis** players. It builds on research in **motion tracking** and **action recognition** to enhance sports performance through a machine learning-based system that analyzes video-captured body movements and provides real-time, actionable insights.

At the core of the system is a **pose estimation module** that tracks key movements—such as the _bandeja_ shot. Video data from both professional and amateur players was collected under controlled conditions, and each frame was annotated to map body joint positions. These mappings were categorized based on the quality of the technique and used to train a supervised recursive model. This model evaluates new player movements by comparing them to ideal movement patterns observed in professionals.

A user-friendly digital interface delivers instant feedback and highlights technique deviations, allowing players to make real-time adjustments and foster structured skill development. This system contributes to the fields of **AI**, **computer vision**, and **sports analytics**, providing a practical tool for improving paddle tennis performance.

---

## 🛠️ Reproducing the Development Environment

To replicate the development environment on another machine:

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
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
   If you are using `requirements.txt` for pip-only dependencies:
   ```bash
   pip install -r requirements.txt
   ```

This ensures that all dependencies and versions used in development are reproduced accurately.

---

## 📁 Project Structure

```
THESISREPO_A00830006/
│
├── Results/
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
├── testVideos/
│
├── environment.yml
├── requirements.txt
└── README.md
```

- `Source/Modules/`: Contains all functional Python modules for data processing, model training/testing, pose estimation, and visualization.
- `Source/main.py`: The main script to execute the system pipeline using command-line arguments.
- `openPoseRequirements/`: Contains setup and dependencies related to OpenPose.
- `Results/`: Stores output data such as joint coordinates.
- `testVideos/`: Directory for input test videos.
- `environment.yml`: Conda environment specification.
- `requirements.txt`: Additional Python dependencies (pip-based).
- `README.md`: Project documentation (you are here!).

---

## 🚀 Running the Code

The `main.py` script uses a command-line interface via `argparse` to allow modular execution of the system components. You can run specific tasks such as pose extraction or coordinate plotting by providing the appropriate command-line flags.

### 🔧 Example Commands

**To extract pose data from a test video using OpenPose:**
```bash
cd Source
python main.py pose --camera ..\testVideos\<video_filename>.mp4
```

This will process the input video and save the extracted joint coordinates (shoulder, elbow, wrist) to a Python file inside the `Results/` directory.

---

**To visualize a coordinate file (previously extracted):**
```bash
cd Source
python main.py plot --file ..\Results\<coordinate_filename>.py
```

This generates a visual representation of the movement trajectory from the coordinate file. It is useful for comparing technical patterns and analyzing shot quality.

> More commands (e.g., training or testing the LSTM model) will be added as development progresses.

---

## 👤 Authors

**Cesar Emilio Castaño Marin**  
Master’s Thesis Student – ITESM  

**Marcial Roberto Leyva Fernández**  
Thesis Advisor  
School of Engineering and Sciences – Tecnológico de Monterrey
        
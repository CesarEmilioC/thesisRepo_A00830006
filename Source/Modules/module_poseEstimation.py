"""
module_poseEstimation.py
========================

This module provides two main functionalities:

1. **Pose Estimation Pipeline**
   - Extracts 2D coordinates for pelvis, right elbow, and right wrist from video clips
     using the TensorFlow-based OpenPose implementation.
   - Saves extracted coordinates, metadata, timestamps, and relative positions in a
     structured JSON format.
   - Designed specifically for paddle tennis “bandeja” shot analysis.

2. **JSON Coordinate Quality Analysis**
   - Computes the ratio of valid pose detections (frames where all 3 keypoints were found)
     relative to total video frames.
   - Can analyze either a single JSON file or recursively traverse directories to analyze
     full coordinate datasets.
   - Used to quantify pose estimation reliability and dataset quality.

This module is invoked through `main.py` via subcommands:
- `pose` → executes pose estimation
- `analyzeJSON` → evaluates JSON coordinate completeness

Author: Cesar Castaño  
Documented by: ChatGPT  
"""

import argparse
import logging
import time
import json
import os
import cv2
import datetime
import re

from openPoseRequirements.tf_pose.estimator import TfPoseEstimator
from openPoseRequirements.tf_pose.networks import get_graph_path, model_wh

# Absolute path to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Folder where JSON coordinate files will be saved
results_dir = os.path.join(ROOT_DIR, 'Coordinates')


# ============================================================================================
# VIDEO → POSE ESTIMATION → JSON COORDINATES
# ============================================================================================

def analyze_video(video_path, args, show_video=False):
    """
    Run OpenPose on a single video file and save extracted coordinates as JSON.

    This function:
    ----------------
    - Loads the specified video from disk.
    - Initializes TensorFlow-based OpenPose using the configured model.
    - Iterates through each frame and extracts pelvis, right elbow, and right wrist.
    - Stores absolute and pelvis-relative coordinates.
    - Builds a structured JSON file with metadata + coordinates.
    - (Optional) displays processed frames with drawn keypoints.

    Parameters
    ----------
    video_path : str
        Path to the .mp4 file to analyze.

    args : argparse.Namespace
        Contains OpenPose configuration parameters (model, resize, resize_out_ratio, etc.).

    show_video : bool, optional
        If True, displays the video while processing. Defaults to False.

    Notes
    -----
    - Frames where *any* of the 3 key joints are missing are discarded.
    - Coordinates are inverted in Y to match a standard mathematical coordinate system.
    - JSONs are always saved inside `Coordinates/` folder.
    """

    logger = logging.getLogger('TfPoseEstimator')
    fps_time = 0

    # Initialize storage dictionary
    coords = {
        "metadata": {
            "video_path": None,
            "video_name": None,
            "player_id": None,
            "part": None,
            "clip": None,
            "grade": None,
            "duration": None,
            "fps": None,
            "frame_width": None,
            "frame_height": None,
            "total_frames": 0,
            "model": args.model,
            "resize": args.resize,
            "resize_out_ratio": args.resize_out_ratio,
            "reference_point": "Pelvis",
            "keypoints_used": ["Pelvis", "Right Elbow", "Right Wrist"],
            "processing_date": datetime.datetime.now().isoformat()
        },
        "Pelvis": [],
        "Mano Derecha Original": [],
        "Mano Derecha Referencia": [],
        "Codo Derecha Original": [],
        "Codo Derecha Referencia": [],
        "timestamps": [],
        "total_coordinates": 0
    }

    # Initialize OpenPose model
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()

    # --- Extract metadata from video ---
    fps = cam.get(cv2.CAP_PROP_FPS)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else None

    coords["metadata"]["fps"] = fps
    coords["metadata"]["duration"] = duration
    coords["metadata"]["frame_width"] = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    coords["metadata"]["frame_height"] = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    coords["metadata"]["total_frames"] = total_frames

    if show_video:
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)

    # =====================================================================================
    # FRAME LOOP
    # =====================================================================================
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        timestamp_sec = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        humans = estimator.inference(frame, resize_to_default=(w > 0 and h > 0),
                                     upsample_size=args.resize_out_ratio)

        # Select target human (pelvis with largest x)
        if humans:
            selected_human = None
            max_x = -1e9

            for hmn in humans:
                pelvis = hmn.body_parts.get(8)
                if pelvis:
                    px = int(pelvis.x * frame.shape[1])
                    if px > max_x:
                        max_x = px
                        selected_human = hmn

            if selected_human:
                pelvis = selected_human.body_parts.get(8)
                elbow = selected_human.body_parts.get(3)
                wrist = selected_human.body_parts.get(4)

                if pelvis and elbow and wrist:

                    # Absolute pixel coordinates
                    H = frame.shape[0]
                    pelvis_abs = (int(pelvis.x * frame.shape[1]),
                                  int((1 - pelvis.y) * H))
                    wrist_abs = (int(wrist.x * frame.shape[1]),
                                 int((1 - wrist.y) * H))
                    elbow_abs = (int(elbow.x * frame.shape[1]),
                                 int((1 - elbow.y) * H))

                    # Pelvis-relative coordinates
                    wrist_rel = (wrist_abs[0] - pelvis_abs[0],
                                 wrist_abs[1] - pelvis_abs[1])
                    elbow_rel = (elbow_abs[0] - pelvis_abs[0],
                                 elbow_abs[1] - pelvis_abs[1])

                    # Store coordinates
                    coords["Pelvis"].append(pelvis_abs)
                    coords["Mano Derecha Original"].append(wrist_abs)
                    coords["Mano Derecha Referencia"].append(wrist_rel)
                    coords["Codo Derecha Original"].append(elbow_abs)
                    coords["Codo Derecha Referencia"].append(elbow_rel)
                    coords["timestamps"].append(timestamp_sec)
                    coords["total_coordinates"] += 1

        if show_video:
            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) == 27:
                break

    cam.release()
    if show_video:
        cv2.destroyAllWindows()

    # =====================================================================================
    # POST-PROCESSING (metadata, JSON generation)
    # =====================================================================================

    video_name = os.path.basename(video_path).replace(".mp4", "")
    coords["metadata"]["video_name"] = video_name

    # Extract identifiers using regex naming convention
    match = re.match(r'player(\d+)_part(\d+)_clip(\d+)_grade(\d+)', video_name)
    if match:
        coords["metadata"]["player_id"] = int(match.group(1))
        coords["metadata"]["part"] = int(match.group(2))
        coords["metadata"]["clip"] = int(match.group(3))
        coords["metadata"]["grade"] = int(match.group(4))

    # Normalize path for JSON
    coords["metadata"]["video_path"] = video_path.replace("\\", "/")

    # Save JSON result
    save_path = os.path.join(results_dir, f"{video_name}.json")
    with open(save_path, "w") as jf:
        json.dump(coords, jf, indent=4)

    print(f"[INFO] Saved coordinates JSON → {save_path}")

def run_json_analysis(args):
    """
    Analiza uno o varios JSONs de coordenadas para obtener la proporción
    de frames válidos (= total_coordinates / total_frames).

    Parámetros esperados:
    - args.file      -> ruta a un solo JSON
    - args.directory -> ruta a un directorio con JSONs (se busca recursivamente)
    """

    # ----------------------------------------
    # CASO 1: Analizar un solo archivo
    # ----------------------------------------
    if args.file:
        if not os.path.isfile(args.file):
            print(f"[ERROR] El archivo no existe: {args.file}")
            return

        ratio = analyze_single_json(args.file)
        if ratio is not None:
            print(f"\n📄 Archivo analizado: {args.file}")
            print(f"➡️  Porcentaje de frames válidos: {ratio*100:.2f}%")
        return

    # ----------------------------------------
    # CASO 2: Analizar un directorio completo
    # ----------------------------------------
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"[ERROR] El directorio no existe: {args.directory}")
            return

        print(f"\n🔍 Buscando JSONs en: {args.directory}\n")

        json_files = []
        for root, _, files in os.walk(args.directory):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))

        if not json_files:
            print("[ERROR] No se encontraron archivos JSON.")
            return

        ratios = []
        for path in json_files:
            r = analyze_single_json(path)
            if r is not None:
                ratios.append(r)
                print(f"{os.path.basename(path)} -> {r*100:.2f}%")

        if ratios:
            avg = sum(ratios) / len(ratios)
            print("\n===================================")
            print("📊 RESUMEN FINAL")
            print("===================================")
            print(f"JSONs analizados: {len(ratios)}")
            print(f"Promedio porcentaje válido: {avg*100:.2f}%")
        return

    print("[ERROR] Debes usar --file o --directory")


def analyze_single_json(path):
    """
    Analiza un solo JSON y retorna el ratio de frames válidos.
    """

    try:
        with open(path, "r") as f:
            data = json.load(f)

        total_frames = data["metadata"]["total_frames"]
        total_coordinates = data.get("total_coordinates", None)

        if total_coordinates is None:
            print(f"[WARN] JSON sin campo total_coordinates: {path}")
            return None

        if total_frames <= 0:
            print(f"[WARN] total_frames inválido en {path}")
            return None

        ratio = total_coordinates / total_frames
        return ratio

    except Exception as e:
        print(f"[ERROR] No se pudo analizar {path}: {e}")
        return None

# ============================================================================================
# Pose estimation execution wrapper (called from main.py)
# ============================================================================================

def run_pose_estimation(args):
    """
    Wrapper for executing full pose estimation on:
    - a single video (--camera), or
    - a directory with multiple .mp4 files (--directory)

    Parameters
    ----------
    args : argparse.Namespace
        Contains OpenPose configuration and file paths.

    Notes
    -----
    - Does NOT handle JSON analysis.
    - Only performs raw pose estimation and JSON generation.
    """

    show_video = bool(args.show_video)

    # Single-file mode
    if args.camera != "0":
        return analyze_video(args.camera, args, show_video)

    # Directory mode
    elif args.directory != "0":
        print("Processing directory:", args.directory)
        for fname in os.listdir(args.directory):
            if fname.lower().endswith(".mp4"):
                video_path = os.path.join(args.directory, fname)
                analyze_video(video_path, args, show_video)
        return

    print("Please provide either --camera <video.mp4> or --directory <folder_path>.")
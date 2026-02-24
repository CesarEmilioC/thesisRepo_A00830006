"""
module_poseEstimation.py
========================

This module provides two main functionalities:

1. **Pose Estimation Pipeline**
   - Extracts 2D coordinates for pelvis, left shoulder, right elbow, and right wrist
     from video clips using the TensorFlow-based OpenPose implementation.
   - For frames where some keypoints are missing but the pelvis is detected, the missing
     values are recorded as NaN and later filled via linear interpolation.
   - Saves extracted coordinates, metadata, timestamps, and relative positions in a
     structured JSON format.
   - Designed specifically for paddle tennis "bandeja" shot analysis.

2. **JSON Coordinate Quality Analysis**
   - Computes the ratio of valid pose detections relative to total video frames.
   - Can analyze either a single JSON file or recursively traverse directories to analyze
     full coordinate datasets.
   - Used to quantify pose estimation reliability and dataset quality.

This module is invoked through `main.py` via subcommands:
- `pose` -> executes pose estimation
- `analyzeJSON` -> evaluates JSON coordinate completeness

Author: Cesar Castaño
Documented by: ChatGPT
"""

import argparse
import logging
import json
import os
import cv2
import datetime
import re
import numpy as np
from typing import Optional, List, Tuple

import config
from openPoseRequirements.tf_pose.estimator import TfPoseEstimator
from openPoseRequirements.tf_pose.networks import get_graph_path, model_wh


# ============================================================================================
# VIDEO -> POSE ESTIMATION -> JSON COORDINATES
# ============================================================================================

def interpolate_coordinates(coord_list: List[Tuple[float, float]]) -> Tuple[List[Tuple[int, int]], int]:
    """Linearly interpolate NaN values in a list of (x, y) coordinate tuples.

    Parameters
    ----------
    coord_list : list of (float, float)
        Coordinate pairs where NaN indicates missing detections.

    Returns
    -------
    (interpolated_list, count) : (list of (int, int), int)
        The interpolated coordinate list and the number of frames that were interpolated.
    """
    arr = np.array(coord_list, dtype=float)
    interpolated_count = 0

    for axis in range(2):  # X=0, Y=1
        values = arr[:, axis]
        valid = ~np.isnan(values)
        if valid.sum() < 2:
            # Not enough valid points to interpolate -- leave as-is
            continue
        invalid = np.isnan(values)
        if invalid.any():
            interpolated_count += int(invalid.sum())
            values[invalid] = np.interp(
                np.where(invalid)[0],
                np.where(valid)[0],
                values[valid]
            )
            arr[:, axis] = values

    result = [(int(round(x)), int(round(y))) for x, y in arr]
    # Each frame with NaN is counted once (not per-axis), so divide by 2
    return result, interpolated_count // 2


def _init_estimator(args: argparse.Namespace):
    """Initialize and return a TfPoseEstimator instance (call once, reuse for many clips)."""
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        return TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    return TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))


def analyze_video(video_path: str, args: argparse.Namespace, show_video: bool = False,
                  estimator=None) -> None:
    """
    Run OpenPose on a single video file and save extracted coordinates as JSON.

    This function:
    ----------------
    - Loads the specified video from disk.
    - Runs OpenPose inference using the provided (or newly created) estimator.
    - Iterates through each frame and extracts pelvis, left shoulder, right elbow,
      and right wrist keypoints.
    - For frames where the pelvis is detected but other keypoints are missing, NaN
      values are recorded and later filled via linear interpolation.
    - Frames where the pelvis is NOT detected are skipped entirely.
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

    estimator : TfPoseEstimator, optional
        Pre-initialized estimator to reuse across clips. If None, a new one is created.

    Notes
    -----
    - Coordinates are inverted in Y to match a standard mathematical coordinate system.
    - JSONs are always saved inside `Coordinates/` folder.
    - Missing keypoints (except pelvis) are interpolated linearly between valid frames.
    """

    # Check if JSON already exists for this clip -- skip if so
    video_name = os.path.basename(video_path).replace(".mp4", "")
    match_check = re.match(config.FILENAME_REGEX, video_name)
    if match_check:
        expected_path = os.path.join(
            config.COORDINATES_DIR,
            f"player{match_check.group(1)}",
            f"part{match_check.group(2)}",
            f"{video_name}.json"
        )
    else:
        expected_path = os.path.join(config.COORDINATES_DIR, f"{video_name}.json")

    if os.path.isfile(expected_path):
        print(f"[SKIP] Coordinate file already exists: {expected_path}")
        return

    logger = logging.getLogger('TfPoseEstimator')

    # Initialize OpenPose model only if not provided
    if estimator is None:
        estimator = _init_estimator(args)

    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return
    ret, frame = cam.read()

    # --- Extract metadata from video ---
    fps = cam.get(cv2.CAP_PROP_FPS)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else None
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if show_video:
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)

    # =====================================================================================
    # FRAME LOOP -- collect raw detections (NaN for missing keypoints)
    # =====================================================================================
    raw_pelvis = []
    raw_shoulder = []
    raw_elbow = []
    raw_wrist = []
    raw_timestamps = []

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        timestamp_sec = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rw, rh = model_wh(args.resize)
        humans = estimator.inference(frame, resize_to_default=(rw > 0 and rh > 0),
                                     upsample_size=args.resize_out_ratio)

        # Select target human: choose the person with the rightmost pelvis position.
        # This heuristic works because the camera is positioned to the left of the
        # court, so the target player (performing the bandeja) is always the rightmost
        # person in the frame.
        if humans:
            selected_human = None
            max_x = -1e9

            for hmn in humans:
                pelvis = hmn.body_parts.get(config.KEYPOINT_PELVIS)
                if pelvis:
                    px = int(pelvis.x * frame.shape[1])
                    if px > max_x:
                        max_x = px
                        selected_human = hmn

            if selected_human:
                pelvis = selected_human.body_parts.get(config.KEYPOINT_PELVIS)

                # Pelvis is required -- skip frame entirely if not detected
                if pelvis:
                    H = frame.shape[0]
                    W = frame.shape[1]

                    pelvis_abs = (int(pelvis.x * W), int((1 - pelvis.y) * H))

                    # Extract other keypoints (NaN if not detected)
                    shoulder = selected_human.body_parts.get(config.KEYPOINT_LEFT_SHOULDER)
                    elbow = selected_human.body_parts.get(config.KEYPOINT_RIGHT_ELBOW)
                    wrist = selected_human.body_parts.get(config.KEYPOINT_RIGHT_WRIST)

                    shoulder_abs = (int(shoulder.x * W), int((1 - shoulder.y) * H)) if shoulder else (float('nan'), float('nan'))
                    elbow_abs = (int(elbow.x * W), int((1 - elbow.y) * H)) if elbow else (float('nan'), float('nan'))
                    wrist_abs = (int(wrist.x * W), int((1 - wrist.y) * H)) if wrist else (float('nan'), float('nan'))

                    raw_pelvis.append(pelvis_abs)
                    raw_shoulder.append(shoulder_abs)
                    raw_elbow.append(elbow_abs)
                    raw_wrist.append(wrist_abs)
                    raw_timestamps.append(timestamp_sec)

        if show_video:
            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) == 27:
                break

    cam.release()
    if show_video:
        cv2.destroyAllWindows()

    # =====================================================================================
    # POST-PROCESSING: Interpolation + Relative coordinates + JSON generation
    # =====================================================================================

    total_detected = len(raw_pelvis)
    if total_detected == 0:
        print(f"[WARNING] No pelvis detections in {video_path}. Skipping.")
        return

    # Interpolate missing shoulder, elbow, and wrist coordinates
    shoulder_interp, n_interp_shoulder = interpolate_coordinates(raw_shoulder)
    elbow_interp, n_interp_elbow = interpolate_coordinates(raw_elbow)
    wrist_interp, n_interp_wrist = interpolate_coordinates(raw_wrist)
    total_interpolated = n_interp_shoulder + n_interp_elbow + n_interp_wrist

    # Compute pelvis-relative coordinates
    shoulder_rel = []
    elbow_rel = []
    wrist_rel = []
    for i in range(total_detected):
        px, py = raw_pelvis[i]
        shoulder_rel.append((shoulder_interp[i][0] - px, shoulder_interp[i][1] - py))
        elbow_rel.append((elbow_interp[i][0] - px, elbow_interp[i][1] - py))
        wrist_rel.append((wrist_interp[i][0] - px, wrist_interp[i][1] - py))

    # Build metadata
    video_name = os.path.basename(video_path).replace(".mp4", "")

    # Trim video_path to start from "Videos/" if present
    full_path_fwd = video_path.replace("\\", "/")
    videos_idx = full_path_fwd.lower().find("/videos/")
    if videos_idx != -1:
        display_path = full_path_fwd[videos_idx + 1:]  # e.g. "Videos/Clips/player3/part1/..."
    else:
        display_path = full_path_fwd

    metadata = {
        "video_path": display_path,
        "video_name": video_name,
        "player_id": None,
        "part": None,
        "clip": None,
        "grade": None,
        "duration": duration,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "total_frames": total_frames,
        "model": args.model,
        "resize": args.resize,
        "resize_out_ratio": args.resize_out_ratio,
        "reference_point": "Pelvis",
        "keypoints_used": ["Pelvis", "Left Shoulder", "Right Elbow", "Right Wrist"],
        "interpolation_method": "linear",
        "interpolated_frames": total_interpolated,
        "processing_date": datetime.datetime.now().isoformat()
    }

    # Extract identifiers using regex naming convention
    match = re.match(config.FILENAME_REGEX, video_name)
    if match:
        metadata["player_id"] = int(match.group(1))
        metadata["part"] = int(match.group(2))
        metadata["clip"] = int(match.group(3))
        metadata["grade"] = int(match.group(4))
    else:
        print(f"[WARNING] Filename '{video_name}' does not match expected pattern "
              f"'player{{N}}_part{{M}}_clip{{K}}_grade{{G}}'. Metadata fields will be null.")

    # Build final JSON structure
    coords = {
        "metadata": metadata,
        config.FIELD_PELVIS: list(raw_pelvis),
        config.FIELD_SHOULDER_ORIGINAL: [list(p) for p in shoulder_interp],
        config.FIELD_SHOULDER_RELATIVE: [list(p) for p in shoulder_rel],
        config.FIELD_ELBOW_ORIGINAL: [list(p) for p in elbow_interp],
        config.FIELD_ELBOW_RELATIVE: [list(p) for p in elbow_rel],
        config.FIELD_WRIST_ORIGINAL: [list(p) for p in wrist_interp],
        config.FIELD_WRIST_RELATIVE: [list(p) for p in wrist_rel],
        "timestamps": raw_timestamps,
        "total_coordinates": total_detected
    }

    # Save JSON result -- create subdirectories based on player/part if available
    if metadata["player_id"] is not None and metadata["part"] is not None:
        save_dir = os.path.join(
            config.COORDINATES_DIR,
            f"player{metadata['player_id']}",
            f"part{metadata['part']}"
        )
    else:
        save_dir = config.COORDINATES_DIR
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{video_name}.json")
    with open(save_path, "w") as jf:
        json.dump(coords, jf, indent=4)

    print(f"[INFO] Saved coordinates JSON -> {save_path} "
          f"({total_detected} frames, {total_interpolated} interpolated keypoint-frames)")

def run_json_analysis(args: argparse.Namespace) -> None:
    """Analyze one or multiple coordinate JSONs to compute the ratio of valid frames
    (= total_coordinates / total_frames).

    Parameters
    ----------
    args : argparse.Namespace
        - args.file: path to a single JSON file
        - args.directory: path to a directory with JSONs (searched recursively)
    """

    if args.file:
        if not os.path.isfile(args.file):
            print(f"[ERROR] File not found: {args.file}")
            return

        ratio = analyze_single_json(args.file)
        if ratio is not None:
            print(f"\n[INFO] File analyzed: {args.file}")
            print(f"[INFO] Valid frame ratio: {ratio*100:.2f}%")
        return

    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"[ERROR] Directory not found: {args.directory}")
            return

        print(f"\n[INFO] Searching for JSONs in: {args.directory}\n")

        json_files = []
        for root, _, files in os.walk(args.directory):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))

        if not json_files:
            print("[ERROR] No JSON files found.")
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
            print(" SUMMARY")
            print("===================================")
            print(f"JSONs analyzed: {len(ratios)}")
            print(f"Average valid frame ratio: {avg*100:.2f}%")
        return

    print("[ERROR] Please use --file or --directory")


def analyze_single_json(path: str) -> Optional[float]:
    """Analyze a single JSON and return the valid frame ratio."""

    try:
        with open(path, "r") as f:
            data = json.load(f)

        total_frames = data["metadata"]["total_frames"]
        total_coordinates = data.get("total_coordinates", None)

        if total_coordinates is None:
            print(f"[WARNING] JSON missing total_coordinates field: {path}")
            return None

        if total_frames <= 0:
            print(f"[WARNING] Invalid total_frames in {path}")
            return None

        ratio = total_coordinates / total_frames
        return ratio

    except Exception as e:
        print(f"[ERROR] Could not analyze {path}: {e}")
        return None

# ============================================================================================
# Pose estimation execution wrapper (called from main.py)
# ============================================================================================

def run_pose_estimation(args: argparse.Namespace) -> None:
    """
    Wrapper for executing full pose estimation on:
    - a single video (--video), or
    - a directory with multiple .mp4 files (--directory)

    Parameters
    ----------
    args : argparse.Namespace
        Contains OpenPose configuration and file paths.
    """

    show_video = bool(args.show_video)

    # Single-file mode
    if args.video is not None:
        if not os.path.isfile(args.video):
            print(f"[ERROR] Video file not found: {args.video}")
            return
        return analyze_video(args.video, args, show_video)

    # Directory mode -- initialize estimator once and reuse for all clips
    elif args.directory is not None:
        if not os.path.isdir(args.directory):
            print(f"[ERROR] Directory not found: {args.directory}")
            return

        mp4_files = sorted(f for f in os.listdir(args.directory) if f.lower().endswith(".mp4"))
        print(f"[INFO] Processing directory: {args.directory} ({len(mp4_files)} videos found)")

        estimator = _init_estimator(args)
        failed = []

        for fname in mp4_files:
            video_path = os.path.join(args.directory, fname)
            try:
                analyze_video(video_path, args, show_video, estimator=estimator)
            except Exception as e:
                print(f"[ERROR] Failed to process {fname}: {e}")
                failed.append(fname)

        if failed:
            print(f"\n[SUMMARY] {len(failed)} clip(s) failed: {', '.join(failed)}")
        else:
            print(f"\n[SUMMARY] All {len(mp4_files)} clip(s) processed successfully.")
        return

    print("[ERROR] Please provide either --video <video.mp4> or --directory <folder_path>.")
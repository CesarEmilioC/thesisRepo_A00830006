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
# Results folder
results_dir = os.path.join(ROOT_DIR, 'Coordinates')


def analyze_video(video_path, args, show_video=False):
    """
    Analyze a single video file, run pose estimation, and save coordinates JSON.

    The function:
    - Loads the video and pose estimation model.
    - Extracts pelvis, right elbow, and right wrist keypoints (absolute and relative).
    - Builds a metadata dictionary with video info and identifiers parsed from filename.
    - Optionally shows the video with keypoints drawn (disabled by default).
    - Saves the results (metadata + coordinates) into a JSON file.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    args : argparse.Namespace
        Contains model, resize, resize_out_ratio, etc.
    show_video : bool, optional
        If True, displays the video with keypoints while processing (default=False).
    """

    logger = logging.getLogger('TfPoseEstimator')
    fps_time = 0

    # Initialize coordinates and metadata
    coords = {
        "metadata": {
            "video_path": None,       # normalized later
            "video_name": None,       # filled later
            "player_id": None,        # extracted from filename
            "part": None,             # extracted from filename
            "clip": None,             # extracted from filename
            "grade": None,            # extracted from filename
            "duration": None,         # filled later
            "fps": None,              # filled later
            "frame_width": None,      # filled later
            "frame_height": None,     # filled later
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

    # Initialize pose estimator
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    cam = cv2.VideoCapture(video_path)
    ret_val, image = cam.read()

    # Extract video metadata
    fps = cam.get(cv2.CAP_PROP_FPS)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else None
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    coords["metadata"]["fps"] = fps
    coords["metadata"]["frame_width"] = frame_width
    coords["metadata"]["frame_height"] = frame_height
    coords["metadata"]["duration"] = duration
    coords["metadata"]["total_frames"] = total_frames

    logger.info('Analyzing %s (%dx%d, %.2f fps, %.2f s)',
                video_path, frame_width, frame_height, fps, duration or 0)

    if show_video:
        cv2.namedWindow('tf-pose-estimation result', cv2.WINDOW_NORMAL)

    while True:
        ret_val, image = cam.read()
        if not ret_val:
            break

        # Current video time in seconds
        video_time = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0),
                             upsample_size=args.resize_out_ratio)

        if humans:  # At least one person detected
            selected_human = None
            max_x = float('-inf')

            # Select person with pelvis farthest to the right
            for human in humans:
                pelvis = human.body_parts.get(8, None)  # Pelvis (COCO id=8)
                if pelvis:
                    pelvis_x = int(pelvis.x * image.shape[1])
                    if pelvis_x > max_x:
                        max_x = pelvis_x
                        selected_human = human

            if selected_human:
                pelvis = selected_human.body_parts.get(8, None)
                right_elbow = selected_human.body_parts.get(3, None)
                right_wrist = selected_human.body_parts.get(4, None)

                if pelvis and right_wrist and right_elbow:
                    # Absolute pixel coordinates
                    pelvis_coords = (int(pelvis.x * image.shape[1]), int(pelvis.y * image.shape[0]))
                    right_wrist_coords = (int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0]))
                    right_elbow_coords = (int(right_elbow.x * image.shape[1]), int(right_elbow.y * image.shape[0]))

                    if show_video:
                        # Draw keypoints on frame
                        cv2.circle(image, pelvis_coords, 8, (0, 255, 0), -1)
                        cv2.circle(image, right_wrist_coords, 8, (0, 0, 255), -1)
                        cv2.circle(image, right_elbow_coords, 8, (255, 0, 0), -1)

                    # Invert Y coordinates
                    height = image.shape[0]
                    pelvis_coords = (pelvis_coords[0], int((1 - pelvis.y) * height))
                    right_wrist_coords = (right_wrist_coords[0], int((1 - right_wrist.y) * height))
                    right_elbow_coords = (right_elbow_coords[0], int((1 - right_elbow.y) * height))

                    # Relative coordinates to pelvis
                    right_wrist_relative = (right_wrist_coords[0] - pelvis_coords[0],
                                            right_wrist_coords[1] - pelvis_coords[1])
                    right_elbow_relative = (right_elbow_coords[0] - pelvis_coords[0],
                                            right_elbow_coords[1] - pelvis_coords[1])

                    # Store results
                    coords["Pelvis"].append(pelvis_coords)
                    coords["Mano Derecha Original"].append(right_wrist_coords)
                    coords["Mano Derecha Referencia"].append(right_wrist_relative)
                    coords["Codo Derecha Original"].append(right_elbow_coords)
                    coords["Codo Derecha Referencia"].append(right_elbow_relative)
                    coords["timestamps"].append(video_time)
                    coords["total_coordinates"] += 1

        if show_video:
            # Show FPS overlay
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()

            if cv2.waitKey(1) == 27:
                break

    cam.release()
    if show_video:
        cv2.destroyAllWindows()

    # --- Post-processing metadata ---
    # Extract filename
    start = video_path.rfind(os.sep) + 1
    end = video_path.rfind(".mp4")
    video_name = video_path[start:end]
    coords["metadata"]["video_name"] = video_name

    # Extract identifiers: player, part, clip, grade
    # Example: player4_part1_clip30_grade6
    match = re.match(r'player(\d+)_part(\d+)_clip(\d+)_grade(\d+)', video_name)
    if match:
        coords["metadata"]["player_id"] = int(match.group(1))
        coords["metadata"]["part"] = int(match.group(2))
        coords["metadata"]["clip"] = int(match.group(3))
        coords["metadata"]["grade"] = int(match.group(4))

    # Normalize path: keep only from "Videos" onward if present
    if "Videos" in video_path:
        idx = video_path.find("Videos")
        coords["metadata"]["video_path"] = video_path[idx:].replace("\\", "/")
    else:
        coords["metadata"]["video_path"] = video_path.replace("\\", "/")

    # Save results as JSON
    coordinates_dir = os.path.join(results_dir, f'{video_name}.json')
    with open(coordinates_dir, "w") as file:
        json.dump(coords, file, indent=4)

    logger.info("Saved coordinates to %s", coordinates_dir)


def run_pose_estimation(args):
    """
    Runs pose estimation either on a single video (--camera) or all videos in a directory (--directory).
    """

    if args.camera != "0":
        analyze_video(args.camera, args, show_video=False)  # change to True to see video
    elif args.directory != "0":
        print("This is the directory", args.directory)
        for file in os.listdir(args.directory):
            if file.lower().endswith(".mp4"):
                video_path = os.path.join(args.directory, file)
                analyze_video(video_path, args, show_video=False)  # change to True if you want visualization
    else:
        print("Please provide either --camera <video_path> or --directory <folder_path>.")
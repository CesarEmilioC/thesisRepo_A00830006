import argparse
import logging
import time
import json
import os
import cv2
import datetime

from openPoseRequirements.tf_pose.estimator import TfPoseEstimator
from openPoseRequirements.tf_pose.networks import get_graph_path, model_wh

# Absolute path to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Results folder
results_dir = os.path.join(ROOT_DIR, 'Samples/coordinateSamples')


def run_pose_estimation(args):
    """
    Runs pose estimation on a given video input, extracts coordinates of keypoints,
    and saves them into a JSON file with coordinates and metadata.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments including:
        - args.model: Pose estimation model type.
        - args.resize: Image resize option for the model.
        - args.resize_out_ratio: Resize ratio for upsampling.
        - args.camera: Path to video file or camera index.

    Output
    ------
    JSON file containing:
    - Metadata (video info, processing info, pose info).
    - Extracted coordinates with timestamps.
    """

    logger = logging.getLogger('TfPoseEstimator-WebCam')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fps_time = 0

    # Initialize coordinates and metadata container
    coords = {
        "metadata": {
            "video_path": args.camera,
            "video_name": None,       # filled later
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

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
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

    logger.info('cam image=%dx%d' % (frame_width, frame_height))

    cv2.namedWindow('tf-pose-estimation result', cv2.WINDOW_NORMAL)
    while True:
        ret_val, image = cam.read()
        if not ret_val:
            break

        # Current video time in seconds
        video_time = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        if humans:  # At least one person detected
            selected_human = None
            max_x = float('-inf')

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
                    pelvis_coords = (int(pelvis.x * image.shape[1]), int(pelvis.y * image.shape[0]))
                    right_wrist_coords = (int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0]))
                    right_elbow_coords = (int(right_elbow.x * image.shape[1]), int(right_elbow.y * image.shape[0]))

                    # Draw keypoints
                    cv2.circle(image, pelvis_coords, 8, (0, 255, 0), -1)
                    cv2.circle(image, right_wrist_coords, 8, (0, 0, 255), -1)
                    cv2.circle(image, right_elbow_coords, 8, (255, 0, 0), -1)

                    # Invert Y
                    height = image.shape[0]
                    pelvis_coords = (pelvis_coords[0], int((1 - pelvis.y) * height))
                    right_wrist_coords = (right_wrist_coords[0], int((1 - right_wrist.y) * height))
                    right_elbow_coords = (right_elbow_coords[0], int((1 - right_elbow.y) * height))

                    # Relative coords
                    right_wrist_relative = (right_wrist_coords[0] - pelvis_coords[0],
                                            right_wrist_coords[1] - pelvis_coords[1])
                    right_elbow_relative = (right_elbow_coords[0] - pelvis_coords[0],
                                            right_elbow_coords[1] - pelvis_coords[1])

                    coords["Pelvis"].append(pelvis_coords)
                    coords["Mano Derecha Original"].append(right_wrist_coords)
                    coords["Mano Derecha Referencia"].append(right_wrist_relative)
                    coords["Codo Derecha Original"].append(right_elbow_coords)
                    coords["Codo Derecha Referencia"].append(right_elbow_relative)
                    coords["timestamps"].append(video_time)
                    coords["total_coordinates"] += 1

        # Show FPS
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    
    # Save data to JSON
    path = args.camera
    start = path.rfind("\\") + 1
    end = path.rfind(".mp4")
    video_name = path[start:end]
    coords["metadata"]["video_name"] = video_name

    coordinates_dir = os.path.join(results_dir, f'{video_name}.json')
    with open(coordinates_dir, "w") as file:
        json.dump(coords, file, indent=4)
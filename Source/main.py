"""
main.py
=======
CLI entry point for the Paddle Tennis Movement Feedback System.

This script provides a command-line interface to orchestrate all project
functionality through subcommands:

    pose         Run OpenPose pose estimation on video files to extract
                 body joint coordinates (pelvis, left shoulder, right elbow, right wrist).
    plot         Visualize coordinate data from JSON files (original,
                 relative, temporal, 3D, or all plot types).
    animate      Create an animation of the detected movement sequence.
    trainLSTM    Train the Bidirectional LSTM model on coordinate data
                 to predict movement quality grades.
    predictLSTM  Predict the quality grade of a single clip using a
                 trained LSTM model.
    countGrades  Count how many clips exist per grade across the dataset.
    analyzeJSON  Analyze JSON data quality by computing the proportion
                 of valid frames (frames where all joints were detected).

Usage examples:
    python main.py pose --video "../Samples/clipSamples/player10_part1_clip0_grade7.mp4"
    python main.py plot --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --type all
    python main.py trainLSTM --directory "../Coordinates" --run_name Test01 --model_path "Models/lstm_model.h5"
    python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"
    python main.py countGrades --directory "../Coordinates"
    python main.py analyzeJSON --directory "../Coordinates"

Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / Paddle Tennis Movement Feedback System
"""

import sys
import os
import argparse
import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'openPoseRequirements')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Modules')))

import Modules.module_poseEstimation as module_poseEstimation
import Modules.module_grapher as module_grapher
import Modules.module_LSTM as module_LSTM


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate module function."""

    parser = argparse.ArgumentParser(
        prog="Pose Estimation Tool",
        description="Integrated tool for paddle tennis movement analysis using OpenPose and LSTM."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ---------------------------------------------------
    # POSE ESTIMATION
    # ---------------------------------------------------
    subparser_pose = subparsers.add_parser(
        "pose",
        help="Run OpenPose pose estimation on video files.",
        epilog='Example: python main.py pose --video "../Samples/clipSamples/player10_part1_clip0_grade7.mp4"'
    )
    subparser_pose.add_argument("--video", type=str, default=None,
                                help="Path to a single video file (.mp4).")
    subparser_pose.add_argument("--directory", type=str, default=None,
                                help="Path to a directory containing multiple .mp4 files.")
    subparser_pose.add_argument("--resize", type=str, default="0x0",
                                help="Input resolution for OpenPose (e.g., '432x368'). Default: native.")
    subparser_pose.add_argument("--resize-out-ratio", type=float, default=4.0,
                                help="Upsample ratio for OpenPose inference. Default: 4.0.")
    subparser_pose.add_argument("--model", type=str, default="mobilenet_thin",
                                help="OpenPose model type. Default: mobilenet_thin.")
    subparser_pose.add_argument("--show-process", action="store_true",
                                help="Display pose estimation details during processing.")
    subparser_pose.add_argument("--show_video", action="store_true",
                                help="Display video window during processing.")
    subparser_pose.set_defaults(func=module_poseEstimation.run_pose_estimation)

    # ---------------------------------------------------
    # PLOT
    # ---------------------------------------------------
    subparser_plot = subparsers.add_parser(
        "plot",
        help="Visualize coordinate data from a JSON file.",
        epilog='Example: python main.py plot --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --type all'
    )
    subparser_plot.add_argument("--file", type=str, required=True,
                                help="Path to a JSON coordinate file.")
    subparser_plot.add_argument("--type", type=str,
                                choices=["original", "relative", "temporal", "3d", "all"],
                                default="all",
                                help="Type of plot to generate. Default: all.")
    subparser_plot.set_defaults(func=module_grapher.plot_coordinates)

    # ---------------------------------------------------
    # ANIMATION
    # ---------------------------------------------------
    subparser_anim = subparsers.add_parser(
        "animate",
        help="Create an animation of the detected movement sequence.",
        epilog='Example: python main.py animate --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"'
    )
    subparser_anim.add_argument("--file", type=str, required=True,
                                help="Path to a JSON coordinate file.")
    subparser_anim.set_defaults(func=module_grapher.animate_motion)

    # ---------------------------------------------------
    # TRAIN LSTM
    # ---------------------------------------------------
    subparser_train = subparsers.add_parser(
        "trainLSTM",
        help="Train the Bidirectional LSTM model on coordinate data.",
        epilog='Example: python main.py trainLSTM --directory "../Coordinates" --run_name Test01 --model_path "Models/lstm_model.h5"'
    )
    subparser_train.add_argument("--directory", type=str, required=True,
                                 help="Path to directory containing JSON coordinate files.")
    subparser_train.add_argument("--run_name", type=str, required=True,
                                 help="Name for the results folder (inside Results/).")
    subparser_train.add_argument("--model_path", type=str,
                                 default="Models/lstm_model.h5",
                                 help="Path to save the trained model. Default: Models/lstm_model.h5.")
    subparser_train.set_defaults(func=module_LSTM.train_model)

    # ---------------------------------------------------
    # PREDICT LSTM
    # ---------------------------------------------------
    subparser_predict = subparsers.add_parser(
        "predictLSTM",
        help="Predict the quality grade of a single clip.",
        epilog='Example: python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json"'
    )
    subparser_predict.add_argument("--file", type=str, required=True,
                                   help="Path to a JSON coordinate file.")
    subparser_predict.add_argument("--model_path", type=str,
                                   default="Models/lstm_model.h5",
                                   help="Path to the trained LSTM model. Default: Models/lstm_model.h5.")
    subparser_predict.set_defaults(func=module_LSTM.predict_clip)

    # ---------------------------------------------------
    # COUNT GRADES
    # ---------------------------------------------------
    subparser_count = subparsers.add_parser(
        "countGrades",
        help="Count how many clips exist per grade across the dataset.",
        epilog='Example: python main.py countGrades --directory "../Coordinates"'
    )
    subparser_count.add_argument("--directory", type=str, required=True,
                                 help="Path to directory containing JSON coordinate files.")
    subparser_count.set_defaults(func=module_LSTM.count_grades)

    # ---------------------------------------------------
    # ANALYZE JSON
    # ---------------------------------------------------
    subparser_analyze = subparsers.add_parser(
        "analyzeJSON",
        help="Analyze the proportion of valid frames in JSON coordinate files.",
        epilog='Example: python main.py analyzeJSON --directory "../Coordinates"'
    )
    subparser_analyze.add_argument("--file", type=str, default="",
                                   help="Path to a single JSON file to analyze.")
    subparser_analyze.add_argument("--directory", type=str, default="",
                                   help="Path to a directory to analyze recursively.")
    subparser_analyze.set_defaults(func=module_poseEstimation.run_json_analysis)

    # ---------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m\n")
    print("Executing command:", args.command)

    try:
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f"\n\033[0;31m[ERROR] Command '{args.command}' failed: {e}\033[0m")
        raise

    print("\n" + "\033[0;32m" + "[end] " + str(datetime.datetime.now()) + "\033[0m\n")


if __name__ == "__main__":
    main()
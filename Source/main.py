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
    trainLSTM    Train the Bidirectional LSTM model on coordinate data.
    predictLSTM  Predict quality grade using a trained LSTM model.
    trainGRU     Train the Bidirectional GRU model on coordinate data.
    predictGRU   Predict quality grade using a trained GRU model.
    trainTCN     Train the TCN (Temporal Convolutional Network) model.
    predictTCN   Predict quality grade using a trained TCN model.
    countGrades  Count how many clips exist per grade across the dataset.
    analyzeJSON  Analyze JSON data quality by computing the proportion
                 of valid frames (frames where all joints were detected).
    regenPlots          Regenerate all 15 experiment plots with display names.
    labelDist           Generate dataset-wide class distribution histogram.
    spearman            Compute Spearman rank correlation for the 3 final models.
    sysOutput           Generate system output comparison figure (low vs high quality).
    datasetStats        Report dataset statistics (clips per player/class/grade).
    thesisMosaic        Generate dataset mosaic (video frame + trajectories per class).
    thesisTrajectories  Generate labeled trajectory figures for Very Low/Medium/Excellent.
    saveAnimation       Save the motion animation for one clip as an animated GIF.

Usage examples:
    python main.py pose --video "../Samples/clipSamples/player10_part1_clip0_grade7.mp4"
    python main.py plot --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --type all
    python main.py trainLSTM --directory "../Coordinates"
    python main.py predictLSTM --file "../Samples/coordinateSamples/player10_part1_clip0_grade7.json" --model-path "Models/lstm_model.h5"
    python main.py countGrades --directory "../Coordinates"
    python main.py analyzeJSON --directory "../Coordinates"

CLI conventions:
    All multi-word flags use kebab-case (dashes), e.g. --show-video, --model-path,
    --thesis-dir. argparse maps these to underscore attributes internally.

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

try:
    import Modules.module_poseEstimation as module_poseEstimation
    _pose_available = True
except (ImportError, ModuleNotFoundError) as _pose_err:
    class _PoseFallback:
        """Placeholder used when OpenPose dependencies are not installed."""
        @staticmethod
        def run_pose_estimation(args):
            print(f"[ERROR] OpenPose is not available ({_pose_err}). "
                  "Install tf_slim and all openPoseRequirements to use this command.")
        @staticmethod
        def run_json_analysis(args):
            print(f"[ERROR] OpenPose is not available ({_pose_err}).")
    module_poseEstimation = _PoseFallback()
    _pose_available = False

import Modules.module_grapher as module_grapher
import Modules.module_data as module_data
import Modules.module_LSTM as module_LSTM
import Modules.module_GRU as module_GRU
import Modules.module_TCN as module_TCN


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
    subparser_pose.add_argument("--show-video", action="store_true",
                                help="Display video window during processing.")
    subparser_pose.add_argument("--output-dir", type=str, default=None,
                                help=("Directory where the JSON (and optional mosaic) "
                                      "will be saved. If provided, the output is written "
                                      "flat (no player/part subfolders). "
                                      "Default: ../Coordinates/player{N}/part{M}/."))
    subparser_pose.add_argument("--save-frames-mosaic", action="store_true",
                                help=("Save a mosaic figure with up to 5 frames where "
                                      "all 4 joints (pelvis, left shoulder, right elbow, "
                                      "right wrist) were detected, including a color "
                                      "legend. Saved alongside the JSON output."))
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
        epilog='Example: python main.py trainLSTM --directory "../Coordinates"'
    )
    subparser_train.add_argument("--directory", type=str, required=True,
                                 help="Path to directory containing JSON coordinate files.")
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
    subparser_predict.add_argument("--model-path", type=str,
                                   default=None,
                                   help="Path to the trained LSTM model. Default: latest lstmModel in Models/.")
    subparser_predict.set_defaults(func=module_LSTM.predict_clip)

    # ---------------------------------------------------
    # TRAIN GRU
    # ---------------------------------------------------
    subparser_train_gru = subparsers.add_parser(
        "trainGRU",
        help="Train the Bidirectional GRU model on coordinate data.",
        epilog='Example: python main.py trainGRU --directory "../Coordinates" --model-path "Models/gru_model.h5"'
    )
    subparser_train_gru.add_argument("--directory", type=str, required=True,
                                      help="Path to directory containing JSON coordinate files.")
    subparser_train_gru.set_defaults(func=module_GRU.train_model)

    # ---------------------------------------------------
    # PREDICT GRU
    # ---------------------------------------------------
    subparser_predict_gru = subparsers.add_parser(
        "predictGRU",
        help="Predict the quality grade of a single clip using a trained GRU model.",
        epilog='Example: python main.py predictGRU --file "../Samples/coordinateSamples/player3_part1_clip1_grade2.json"'
    )
    subparser_predict_gru.add_argument("--file", type=str, required=True,
                                        help="Path to a JSON coordinate file.")
    subparser_predict_gru.add_argument("--model-path", type=str,
                                        default=None,
                                        help="Path to the trained GRU model. Default: latest gruModel in Models/.")
    subparser_predict_gru.set_defaults(func=module_GRU.predict_clip)

    # ---------------------------------------------------
    # TRAIN TCN
    # ---------------------------------------------------
    subparser_train_tcn = subparsers.add_parser(
        "trainTCN",
        help="Train the TCN (Temporal Convolutional Network) model on coordinate data.",
        epilog='Example: python main.py trainTCN --directory "../Coordinates" --model-path "Models/tcn_model.h5"'
    )
    subparser_train_tcn.add_argument("--directory", type=str, required=True,
                                      help="Path to directory containing JSON coordinate files.")
    subparser_train_tcn.set_defaults(func=module_TCN.train_model)

    # ---------------------------------------------------
    # PREDICT TCN
    # ---------------------------------------------------
    subparser_predict_tcn = subparsers.add_parser(
        "predictTCN",
        help="Predict the quality grade of a single clip using a trained TCN model.",
        epilog='Example: python main.py predictTCN --file "../Samples/coordinateSamples/player3_part1_clip1_grade2.json"'
    )
    subparser_predict_tcn.add_argument("--file", type=str, required=True,
                                        help="Path to a JSON coordinate file.")
    subparser_predict_tcn.add_argument("--model-path", type=str,
                                        default=None,
                                        help="Path to the trained TCN model. Default: latest tcnModel in Models/.")
    subparser_predict_tcn.set_defaults(func=module_TCN.predict_clip)

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
    subparser_count.set_defaults(func=module_data.count_grades)

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
    # REGEN PLOTS
    # ---------------------------------------------------
    subparser_regen = subparsers.add_parser(
        "regenPlots",
        help="Regenerate all 15 experiment plots with English labels and display names.",
        epilog='Example: python main.py regenPlots --directory "../Coordinates"'
    )
    subparser_regen.add_argument("--directory", type=str, required=True,
                                 help="Path to directory containing JSON coordinate files.")
    subparser_regen.add_argument("--thesis-dir", type=str, default=None,
                                 help="Path to thesis document folder. Default: auto-detected.")
    subparser_regen.set_defaults(func=module_grapher.regen_plots)

    # ---------------------------------------------------
    # LABEL DISTRIBUTION
    # ---------------------------------------------------
    subparser_ldist = subparsers.add_parser(
        "labelDist",
        help="Generate dataset-wide class distribution histogram.",
        epilog='Example: python main.py labelDist --directory "../Coordinates"'
    )
    subparser_ldist.add_argument("--directory", type=str, required=True,
                                 help="Path to directory containing JSON coordinate files.")
    subparser_ldist.add_argument("--thesis-dir", type=str, default=None,
                                 help="Path to thesis document folder. Default: auto-detected.")
    subparser_ldist.set_defaults(func=module_grapher.label_dist)

    # ---------------------------------------------------
    # SPEARMAN
    # ---------------------------------------------------
    subparser_spear = subparsers.add_parser(
        "spearman",
        help="Compute Spearman rank correlation for the 3 final models.",
        epilog='Example: python main.py spearman --directory "../Coordinates"'
    )
    subparser_spear.add_argument("--directory", type=str, required=True,
                                 help="Path to directory containing JSON coordinate files.")
    subparser_spear.add_argument("--thesis-dir", type=str, default=None,
                                 help="Path to thesis document folder. Default: auto-detected.")
    subparser_spear.set_defaults(func=module_data.compute_spearman)

    # ---------------------------------------------------
    # SYSTEM OUTPUT FIGURE
    # ---------------------------------------------------
    subparser_sysout = subparsers.add_parser(
        "sysOutput",
        help="Generate system output comparison figure (low vs high quality clip).",
        epilog='Example: python main.py sysOutput --directory "../Coordinates"'
    )
    subparser_sysout.add_argument("--directory", type=str, required=True,
                                  help="Path to directory containing JSON coordinate files.")
    subparser_sysout.add_argument("--thesis-dir", type=str, default=None,
                                  help="Path to thesis document folder. Default: auto-detected.")
    subparser_sysout.set_defaults(func=module_grapher.sys_output)

    # ---------------------------------------------------
    # DATASET STATISTICS
    # ---------------------------------------------------
    subparser_dstats = subparsers.add_parser(
        "datasetStats",
        help="Report dataset statistics (clips per player, class, and grade).",
        epilog='Example: python main.py datasetStats --directory "../Coordinates"'
    )
    subparser_dstats.add_argument("--directory", type=str, required=True,
                                  help="Path to directory containing JSON coordinate files.")
    subparser_dstats.add_argument("--thesis-dir", type=str, default=None,
                                  help="Path to thesis document folder. Default: auto-detected.")
    subparser_dstats.set_defaults(func=module_data.dataset_stats)

    # ---------------------------------------------------
    # THESIS MOSAIC
    # ---------------------------------------------------
    subparser_mosaic = subparsers.add_parser(
        "thesisMosaic",
        help="Generate dataset mosaic: video frame + trajectories per quality class.",
        epilog=('Example: python main.py thesisMosaic '
                '--clips-dir "../Videos/Clips" --coords-dir "../Coordinates"')
    )
    subparser_mosaic.add_argument("--clips-dir", type=str, required=True,
                                  help="Path to directory containing .mp4 video clips.")
    subparser_mosaic.add_argument("--coords-dir", type=str, required=True,
                                  help="Path to directory containing JSON coordinate files.")
    subparser_mosaic.add_argument("--thesis-dir", type=str, default=None,
                                  help="Path to thesis document folder. Default: auto-detected.")
    subparser_mosaic.set_defaults(func=module_grapher.dataset_mosaic)

    # ---------------------------------------------------
    # THESIS TRAJECTORIES
    # ---------------------------------------------------
    subparser_traj = subparsers.add_parser(
        "thesisTrajectories",
        help="Generate labeled trajectory figures for Very Low, Medium, and Excellent clips.",
        epilog='Example: python main.py thesisTrajectories --coords-dir "../Coordinates"'
    )
    subparser_traj.add_argument("--coords-dir", type=str, required=True,
                                help="Path to directory containing JSON coordinate files.")
    subparser_traj.add_argument("--thesis-dir", type=str, default=None,
                                help="Path to thesis document folder. Default: auto-detected.")
    subparser_traj.set_defaults(func=module_grapher.thesis_trajectories)

    # ---------------------------------------------------
    # SAVE ANIMATION GIF
    # ---------------------------------------------------
    subparser_gif = subparsers.add_parser(
        "saveAnimation",
        help="Save the motion animation for one clip as an animated GIF.",
        epilog='Example: python main.py saveAnimation --file "../Coordinates/player10/..."'
    )
    subparser_gif.add_argument("--file", type=str, required=True,
                               help="Path to a JSON coordinate file.")
    subparser_gif.add_argument("--out-dir", type=str, default=None,
                               help="Output directory for the GIF. Default: thesis Methodology/.")
    subparser_gif.add_argument("--thesis-dir", type=str, default=None,
                               help="Path to thesis document folder. Default: auto-detected.")
    subparser_gif.set_defaults(func=module_grapher.save_animation_gif)

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
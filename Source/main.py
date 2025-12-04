import sys
import os
import argparse
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'openPoseRequirements')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Modules')))

import Modules.module_poseEstimation as module_poseEstimation
import Modules.module_grapher as module_grapher
import Modules.module_LSTM as module_LSTM


def main():

    parser = argparse.ArgumentParser(
        prog="Pose Estimation Tool",
        description="Herramienta integral para análisis de movimientos en pádel usando OpenPose y LSTM."
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # ---------------------------------------------------
    # POSE ESTIMATION
    # ---------------------------------------------------
    subparser_pose = subparsers.add_parser("pose", help="Realiza la estimación de pose.")
    subparser_pose.add_argument("--camera", type=str, default="0")
    subparser_pose.add_argument("--directory", type=str, default="0")
    subparser_pose.add_argument("--resize", type=str, default="0x0")
    subparser_pose.add_argument("--resize-out-ratio", type=float, default=4.0)
    subparser_pose.add_argument("--model", type=str, default="mobilenet_thin")
    subparser_pose.add_argument("--show-process", action="store_true")
    subparser_pose.add_argument("--show_video", type=bool, default=False)
    subparser_pose.set_defaults(func=module_poseEstimation.run_pose_estimation)

    # ---------------------------------------------------
    # PLOT
    # ---------------------------------------------------
    subparser_plot = subparsers.add_parser("plot", help="Graficar datos JSON.")
    subparser_plot.add_argument("--file", type=str, required=True)
    subparser_plot.add_argument("--type", type=str,
                                choices=["original", "relative", "temporal", "3d", "all"],
                                default="all")
    subparser_plot.set_defaults(func=module_grapher.plot_coordinates)

    # ---------------------------------------------------
    # ANIMACIÓN
    # ---------------------------------------------------
    subparser_anim = subparsers.add_parser("animate", help="Animación del movimiento.")
    subparser_anim.add_argument("--file", type=str, required=True)
    subparser_anim.set_defaults(func=module_grapher.animate_motion)

    # ---------------------------------------------------
    # TRAIN LSTM
    # ---------------------------------------------------
    subparser_train = subparsers.add_parser("trainLSTM", help="Entrenar el modelo LSTM.")
    subparser_train.add_argument("--directory", type=str, required=True)
    subparser_train.add_argument("--run_name", type=str, required=True,
                                 help="Nombre de la carpeta (dentro de Results/) donde guardar resultados.")
    subparser_train.add_argument("--model_path", type=str,
                                 default="Models/lstm_model.h5",
                                 help="Ruta donde se guardará el modelo entrenado.")
    subparser_train.set_defaults(func=module_LSTM.train_model)

    # ---------------------------------------------------
    # PREDICT LSTM
    # ---------------------------------------------------
    subparser_predict = subparsers.add_parser("predictLSTM", help="Predecir calificación.")
    subparser_predict.add_argument("--file", type=str, required=True)
    subparser_predict.add_argument("--model_path", type=str,
                                   default="Models/lstm_model.h5")
    subparser_predict.set_defaults(func=module_LSTM.predict_clip)

    # ---------------------------------------------------
    # COUNT GRADES
    # ---------------------------------------------------
    subparser_count = subparsers.add_parser("countGrades", help="Cuenta clips por calificación.")
    subparser_count.add_argument("--directory", type=str, required=True)
    subparser_count.set_defaults(func=module_LSTM.count_grades)

    # ---------------------------------------------------
    # ANALYZE JSON
    # ---------------------------------------------------
    subparser_analyze = subparsers.add_parser("analyzeJSON",
                                              help="Analiza frames válidos en JSONs.")
    subparser_analyze.add_argument("--file", type=str, default="")
    subparser_analyze.add_argument("--directory", type=str, default="")
    subparser_analyze.set_defaults(func=module_poseEstimation.run_json_analysis)

    # ---------------------------------------------------
    args = parser.parse_args()

    print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m\n")
    print("Ejecutando comando:", args.command)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

    print("\n" + "\033[0;32m" + "[end] " + str(datetime.datetime.now()) + "\033[0m\n")


if __name__ == "__main__":
    main()
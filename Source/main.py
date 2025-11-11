import sys
import os
import argparse
import datetime

# === Añadir rutas de módulos ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'openPoseRequirements')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Modules')))

import Modules.module_poseEstimation as module_poseEstimation
import Modules.module_grapher as module_grapher
import Modules.module_LSTM as module_LSTM


def main():
    parser = argparse.ArgumentParser("Pose Estimation Tool", description="Herramienta integral para análisis de movimientos en pádel")
    subparsers = parser.add_subparsers(dest="command")

    # ==================================
    # Subparser 1: Pose Estimation
    # ==================================
    subparser_pose = subparsers.add_parser("pose", help="Estimación de pose desde cámara o video")
    subparser_pose.add_argument("--camera", type=str, default="0")
    subparser_pose.add_argument("--directory", type=str, default="0")
    subparser_pose.add_argument("--resize", type=str, default="0x0")
    subparser_pose.add_argument("--resize-out-ratio", type=float, default=4.0)
    subparser_pose.add_argument("--model", type=str, default="mobilenet_thin")
    subparser_pose.add_argument("--show-process", action="store_true")
    subparser_pose.set_defaults(func=module_poseEstimation.run_pose_estimation)

    # ==================================
    # Subparser 2: Plot Coordinates
    # ==================================
    subparser_plot = subparsers.add_parser("plot", help="Graficar coordenadas desde un JSON")
    subparser_plot.add_argument("--file", type=str, required=True)
    subparser_plot.add_argument("--type", type=str, choices=["original", "relative", "temporal", "3d", "all"], default="all")
    subparser_plot.set_defaults(func=module_grapher.plot_coordinates)

    # ==================================
    # Subparser 3: Animar Movimiento
    # ==================================
    subparser_anim = subparsers.add_parser("animate", help="Animar el movimiento del codo y mano derecha")
    subparser_anim.add_argument("--file", type=str, required=True)
    subparser_anim.set_defaults(func=module_grapher.animate_motion)

    # ==================================
    # Subparser 4: Entrenar LSTM
    # ==================================
    subparser_train = subparsers.add_parser("trainLSTM", help="Entrenar modelo LSTM con coordenadas")
    subparser_train.add_argument("--directory", type=str, required=True, help="Ruta a la carpeta 'Coordinates'")
    subparser_train.add_argument("--model-path", type=str, default="Models/lstm_model.h5", help="Ruta para guardar el modelo entrenado")
    subparser_train.set_defaults(func=module_LSTM.train_model)

    # ==================================
    # Subparser 5: Predecir con LSTM
    # ==================================
    subparser_predict = subparsers.add_parser("predictLSTM", help="Predecir calificación de un clip usando LSTM")
    subparser_predict.add_argument("--file", type=str, required=True, help="Archivo JSON de coordenadas")
    subparser_predict.add_argument("--model-path", type=str, default="Models/lstm_model.h5", help="Ruta del modelo entrenado")
    subparser_predict.set_defaults(func=module_LSTM.predict_clip)

    # ==================================
    # Ejecutar comando
    # ==================================
    args = parser.parse_args()
    print("\n\033[0;32m[start] " + str(datetime.datetime.now()) + "\033[0m\n")
    print("Ejecutando comando:", args.command)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

    print("\n\033[0;32m[end] " + str(datetime.datetime.now()) + "\033[0m\n")


if __name__ == "__main__":
    main()
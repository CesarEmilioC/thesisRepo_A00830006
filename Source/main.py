import sys
import os
# Agrega requirements/tf_pose al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'openPoseRequirements')))
# Agrega requirements/tf_pose al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Modules')))

import argparse
import datetime
import Modules.module_poseEstimation as module_poseEstimation
import Modules.module_grapher as module_grapher

def main():
    parser = argparse.ArgumentParser("Pose Estimation Tool", description="Herramienta para estimación de pose y graficación de coordenadas")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser para la estimación de pose
    subparser_pose = subparsers.add_parser("pose", help="Realiza la estimación de pose en tiempo real desde la cámara o desde un video")
    subparser_pose.add_argument("--camera", type=str, default="0", help="Índice de cámara o ruta del video. Default=0 (cámara por defecto)")
    subparser_pose.add_argument("--resize", type=str, default="0x0", help="Tamaño de redimensionamiento para las imágenes (ej. 432x368)")
    subparser_pose.add_argument("--resize-out-ratio", type=float, default=4.0, help="Proporción para redimensionar heatmaps")
    subparser_pose.add_argument("--model", type=str, default="mobilenet_thin", help="Modelo a usar: cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small")
    subparser_pose.add_argument("--show-process", action="store_true", help="Muestra información de procesamiento para debugging")
    subparser_pose.set_defaults(func=module_poseEstimation.run_pose_estimation)

    # Subparser para graficar los datos
    subparser_plot = subparsers.add_parser("plot", help="Genera gráficas a partir de los datos guardados en coordenadas.json")
    subparser_plot.add_argument("--file", type=str, required=True, help="Ruta del archivo JSON con coordenadas (ej. coordenadas.json)")
    subparser_plot.set_defaults(func=module_grapher.plot_coordinates)

    args = parser.parse_args()

    print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n")
    print("Ejecutando comando:", args.command)

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

    print("\n" + "\033[0;32m" + "[end] " + str(datetime.datetime.now()) + "\033[0m" + "\n")


if __name__ == "__main__":
    main()
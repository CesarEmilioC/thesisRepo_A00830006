import sys
import os
import argparse
import datetime

# ===============================================
# Añadir rutas de módulos al PYTHONPATH
# ===============================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'openPoseRequirements')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Modules')))

# ===============================================
# Importar módulos internos
# ===============================================
import Modules.module_poseEstimation as module_poseEstimation
import Modules.module_grapher as module_grapher
import Modules.module_LSTM as module_LSTM


def main():
    """
    Herramienta integral para análisis y evaluación de movimientos en pádel.
    -----------------------------------------------------------
    Este programa permite realizar:
    1. Estimación de pose con OpenPose (pose)
    2. Visualización de coordenadas (plot)
    3. Animación de movimientos (animate)
    4. Entrenamiento del modelo LSTM (trainLSTM)
    5. Predicción de calificaciones con LSTM (predictLSTM)
    """

    parser = argparse.ArgumentParser(
        prog="Pose Estimation Tool",
        description="Herramienta integral para análisis de movimientos en pádel usando OpenPose y LSTM."
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # ==================================================
    # Subparser 1: Estimación de pose (OpenPose)
    # ==================================================
    subparser_pose = subparsers.add_parser(
        "pose",
        help="Realiza la estimación de pose desde cámara o video."
    )
    subparser_pose.add_argument(
        "--camera",
        type=str,
        default="0",
        help=(
            "Índice de cámara o ruta de video.\n"
            "Ejemplo: '--camera 0' usa la cámara web, '--camera video.mp4' usa un archivo."
        )
    )
    subparser_pose.add_argument(
        "--directory",
        type=str,
        default="0",
        help="Ruta a una carpeta con múltiples videos a analizar secuencialmente."
    )
    subparser_pose.add_argument(
        "--resize",
        type=str,
        default="0x0",
        help="Redimensionamiento de entrada (ej. '432x368'). Default: 0x0 (sin cambio)."
    )
    subparser_pose.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="Factor de redimensionamiento de los mapas de salida (heatmaps). Default: 4.0."
    )
    subparser_pose.add_argument(
        "--model",
        type=str,
        default="mobilenet_thin",
        help="Modelo de OpenPose a usar: 'cmu', 'mobilenet_thin', 'mobilenet_v2_large', etc."
    )
    subparser_pose.add_argument(
        "--show-process",
        action="store_true",
        help="Muestra información del proceso de inferencia para debugging."
    )
    subparser_pose.set_defaults(func=module_poseEstimation.run_pose_estimation)

    # ==================================================
    # Subparser 2: Graficar coordenadas (grapher)
    # ==================================================
    subparser_plot = subparsers.add_parser(
        "plot",
        help="Genera gráficas a partir de un archivo JSON con coordenadas."
    )
    subparser_plot.add_argument(
        "--file",
        type=str,
        required=True,
        help="Ruta al archivo JSON con coordenadas generado por la estimación de pose."
    )
    subparser_plot.add_argument(
        "--type",
        type=str,
        choices=["original", "relative", "temporal", "3d", "all"],
        default="all",
        help=(
            "Tipo de gráfica a generar:\n"
            "  'original' → coordenadas crudas\n"
            "  'relative' → posiciones relativas a una articulación de referencia\n"
            "  'temporal' → evolución de las coordenadas en el tiempo\n"
            "  '3d' → representación tridimensional\n"
            "  'all' → genera todas las anteriores"
        )
    )
    subparser_plot.set_defaults(func=module_grapher.plot_coordinates)

    # ==================================================
    # Subparser 3: Animar movimiento
    # ==================================================
    subparser_anim = subparsers.add_parser(
        "animate",
        help="Genera una animación del movimiento (codo y mano derecha) desde un JSON."
    )
    subparser_anim.add_argument(
        "--file",
        type=str,
        required=True,
        help="Ruta al archivo JSON con las coordenadas del clip a animar."
    )
    subparser_anim.set_defaults(func=module_grapher.animate_motion)

    # ==================================================
    # Subparser 4: Entrenar modelo LSTM
    # ==================================================
    subparser_train = subparsers.add_parser(
        "trainLSTM",
        help="Entrena el modelo LSTM usando coordenadas en formato JSON."
    )
    subparser_train.add_argument(
        "--directory",
        type=str,
        required=True,
        help=(
            "Ruta a la carpeta principal 'Coordinates'.\n"
            "La estructura debe contener subcarpetas por jugador y parte:\n"
            "  Coordinates/\n"
            "  ├── player1/part1/*.json\n"
            "  ├── player2/part2/*.json ..."
        )
    )
    subparser_train.add_argument(
        "--model-path",
        type=str,
        default="Models/lstm_model.h5",
        help="Ruta donde se guardará el modelo entrenado. Default: 'Models/lstm_model.h5'"
    )
    subparser_train.set_defaults(func=module_LSTM.train_model)

    # ==================================================
    # Subparser 5: Predecir calificación con LSTM
    # ==================================================
    subparser_predict = subparsers.add_parser(
        "predictLSTM",
        help="Predice la calificación de un clip individual usando un modelo LSTM entrenado."
    )
    subparser_predict.add_argument(
        "--file",
        type=str,
        required=True,
        help="Archivo JSON de coordenadas (formato compatible con los usados para entrenamiento)."
    )
    subparser_predict.add_argument(
        "--model-path",
        type=str,
        default="Models/lstm_model.h5",
        help="Ruta al modelo LSTM previamente entrenado. Default: 'Models/lstm_model.h5'"
    )
    subparser_predict.set_defaults(func=module_LSTM.predict_clip)
    
    # ==================================
    # Subparser 6: Contar grades
    # ==================================
    subparser_count = subparsers.add_parser(
        "countGrades",
        help="Muestra cuántos clips hay por calificación"
    )
    subparser_count.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Carpeta con archivos JSON de coordenadas"
    )
    subparser_count.set_defaults(func=module_LSTM.count_grades)


    # ==================================================
    # Ejecución del comando seleccionado
    # ==================================================
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
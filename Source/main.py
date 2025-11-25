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
    Comandos disponibles:
    1. pose          → Estimación de pose con OpenPose
    2. plot          → Gráficas a partir del JSON
    3. animate       → Animación del movimiento
    4. trainLSTM     → Entrenar modelo LSTM
    5. predictLSTM   → Predecir calificación
    6. countGrades   → Contar cuántos clips hay por calificación
    7. analyzeJSON   → Analizar proporción de frames válidos en JSONs
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
    subparser_pose.add_argument("--camera", type=str, default="0",
                                help="Índice de cámara o ruta de video.")
    subparser_pose.add_argument("--directory", type=str, default="0",
                                help="Carpeta con múltiples videos a analizar.")
    subparser_pose.add_argument("--resize", type=str, default="0x0",
                                help="Tamaño de entrada (ej. '432x368').")
    subparser_pose.add_argument("--resize-out-ratio", type=float, default=4.0,
                                help="Escala de los mapas de salida.")
    subparser_pose.add_argument("--model", type=str, default="mobilenet_thin",
                                help="Modelo OpenPose ('cmu', 'mobilenet_thin', etc.)")
    subparser_pose.add_argument("--show-process", action="store_true",
                                help="Muestra información del proceso.")
    subparser_pose.add_argument("--show_video", type=bool, default=False,
                                help="Mostrar el video durante la estimación.")
    subparser_pose.set_defaults(func=module_poseEstimation.run_pose_estimation)

    # ==================================================
    # Subparser 2: Graficar coordenadas (grapher)
    # ==================================================
    subparser_plot = subparsers.add_parser(
        "plot",
        help="Genera gráficas a partir de un archivo JSON."
    )
    subparser_plot.add_argument("--file", type=str, required=True,
                                help="Ruta al archivo JSON.")
    subparser_plot.add_argument("--type", type=str,
                                choices=["original", "relative", "temporal", "3d", "all"],
                                default="all",
                                help="Tipo de gráfica.")
    subparser_plot.set_defaults(func=module_grapher.plot_coordinates)

    # ==================================================
    # Subparser 3: Animación del movimiento
    # ==================================================
    subparser_anim = subparsers.add_parser(
        "animate",
        help="Genera una animación desde un JSON."
    )
    subparser_anim.add_argument("--file", type=str, required=True,
                                help="Archivo JSON de coordenadas.")
    subparser_anim.set_defaults(func=module_grapher.animate_motion)

    # ==================================================
    # Subparser 4: Entrenar modelo LSTM
    # ==================================================
    subparser_train = subparsers.add_parser(
        "trainLSTM",
        help="Entrena el modelo LSTM usando archivos JSON."
    )
    subparser_train.add_argument("--directory", type=str, required=True,
                                 help="Carpeta principal 'Coordinates'.")
    subparser_train.add_argument("--model-path", type=str,
                                 default="Models/lstm_model.h5",
                                 help="Ruta donde se guardará el modelo.")
    subparser_train.set_defaults(func=module_LSTM.train_model)

    # ==================================================
    # Subparser 5: Predecir calificación LSTM
    # ==================================================
    subparser_predict = subparsers.add_parser(
        "predictLSTM",
        help="Predice la calificación de un clip usando un modelo LSTM."
    )
    subparser_predict.add_argument("--file", type=str, required=True,
                                   help="Archivo JSON con coordenadas.")
    subparser_predict.add_argument("--model-path", type=str,
                                   default="Models/lstm_model.h5",
                                   help="Ruta al modelo entrenado.")
    subparser_predict.set_defaults(func=module_LSTM.predict_clip)

    # ==================================================
    # Subparser 6: Contar cuántos clips hay por calificación
    # ==================================================
    subparser_count = subparsers.add_parser(
        "countGrades",
        help="Cuenta la cantidad de clips por calificación."
    )
    subparser_count.add_argument("--directory", type=str, required=True,
                                 help="Carpeta con JSONs.")
    subparser_count.set_defaults(func=module_LSTM.count_grades)

    # ==================================================
    # Subparser 7: ANALIZAR PORCENTAJE DE FRAMES VÁLIDOS
    # ==================================================
    subparser_analyze = subparsers.add_parser(
        "analyzeJSON",
        help="Analiza archivos JSON para calcular la proporción de frames válidos."
    )
    subparser_analyze.add_argument(
        "--file",
        type=str,
        default="",
        help="Ruta a un archivo JSON para analizar."
    )
    subparser_analyze.add_argument(
        "--directory",
        type=str,
        default="",
        help="Ruta a un directorio. Se analizarán TODOS los JSONs recursivamente."
    )
    subparser_analyze.set_defaults(func=module_poseEstimation.run_json_analysis)

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
import argparse
import logging
import time
import json
import os
import cv2

from openPoseRequirements.tf_pose.estimator import TfPoseEstimator
from openPoseRequirements.tf_pose.networks import get_graph_path, model_wh

# Ruta absoluta al directorio raíz del proyecto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Ruta a la carpeta Results
results_dir = os.path.join(ROOT_DIR, 'Results')


def run_pose_estimation(args):

    logger = logging.getLogger('TfPoseEstimator-WebCam')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fps_time = 0

    coords = {
        "Nuca": [],
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
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # Set the video capture to get the FPS
    fps = cam.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow('tf-pose-estimation result', cv2.WINDOW_NORMAL)
    while True:
        ret_val, image = cam.read()
        if not ret_val:
            break

        # Obtener el tiempo actual en el video en milisegundos
        video_time = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convertir a segundos

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        if humans:  # Verifica si hay al menos una persona detectada
            # Filtrar la persona con el cuello (neck) más bajo en la imagen
            selected_human = None
            min_x = float('inf')  # Inicializar con un valor grande

            for human in humans:
                neck = human.body_parts.get(1, None)  # Punto clave del cuello
                if neck:
                    neck_x = int(neck.x * image.shape[0])  # Convertir a coordenadas de píxeles
                    if neck_x < min_x:  # Tomar la persona con el cuello más a la izquierda en la imagen
                        min_x = neck_x
                        selected_human = human

            if selected_human:
                # Extraer puntos clave
                neck = selected_human.body_parts.get(1, None)  # Nuca
                right_elbow = selected_human.body_parts.get(3, None)  # Mano derecha
                right_wrist = selected_human.body_parts.get(4, None)  # Mano derecha

                if neck and right_wrist:
                    # Convertir coordenadas normalizadas a píxeles
                    neck_coords = (int(neck.x * image.shape[1]), int(neck.y * image.shape[0]))
                    right_wrist_coords = (int(right_wrist.x * image.shape[1]), int(right_wrist.y * image.shape[0]))
                    right_elbow_coords = (int(right_elbow.x * image.shape[1]), int(right_wrist.y * image.shape[0]))

                    # Dibujar solo los dos puntos clave
                    cv2.circle(image, neck_coords, 8, (0, 255, 0), -1)  # Verde (Nuca)
                    cv2.circle(image, right_wrist_coords, 8, (0, 0, 255), -1)  # Rojo (Mano Derecha)
                    cv2.circle(image, right_elbow_coords, 8, (255, 0, 0), -1)  # Azul (Codo Derecha)
                    
                    # Invertir la coordenada Y
                    height = image.shape[0]  # Altura de la imagen
                    neck_coords = (int(neck.x * image.shape[1]), int((1 - neck.y) * height))  # Invertir Y
                    right_wrist_coords = (int(right_wrist.x * image.shape[1]), int((1 - right_wrist.y) * height))  # Invertir Y
                    right_elbow_coords = (int(right_elbow.x * image.shape[1]), int((1 - right_elbow.y) * height))  # Invertir Y

                    # Coordenadas relativas
                    right_wrist_relative = (right_wrist_coords[0] - neck_coords[0], right_wrist_coords[1] - neck_coords[1])
                    right_elbow_relative = (right_elbow_coords[0] - neck_coords[0], right_elbow_coords[1] - neck_coords[1])


                    # Guardar las coordenadas con un único timestamp para este ciclo
                    coords["Nuca"].append(neck_coords)
                    coords["Mano Derecha Original"].append(right_wrist_coords)
                    coords["Mano Derecha Referencia"].append(right_wrist_relative)
                    coords["Codo Derecha Original"].append(right_elbow_coords)
                    coords["Codo Derecha Referencia"].append(right_elbow_relative)
                    coords["timestamps"].append(video_time)  # Un único timestamp para todo el ciclo

                    # Incrementar el contador de coordenadas
                    coords["total_coordinates"] += 1

        # Mostrar FPS
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    
    # Guardar datos en JSON
    path = args.camera
    start = path.rfind("\\") + 1
    end = path.rfind(".mp4")
    video_name = path[start:end]
    coordinates_dir = os.path.join(results_dir, f'coordenadas_{video_name}.json')

    with open(coordinates_dir, "w") as file:
        json.dump(coords, file, indent=4)
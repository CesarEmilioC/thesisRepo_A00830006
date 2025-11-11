"""
-------------------------------------------------------------
Module: module_LSTM.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / LSTM for Paddle Tennis Analysis
-------------------------------------------------------------

Descripción:
------------
Este módulo implementa un modelo LSTM (Long Short-Term Memory)
para predecir la calificación (de 1 a 10) de un movimiento de
"bandeja" en pádel a partir de las coordenadas extraídas por
pose estimation.

Cada clip de movimiento se almacena como un archivo JSON con:
    - Coordenadas de Pelvis, Codo Derecho y Mano Derecha
    - Calificación (grade)
    - Metadatos del video

El modelo analiza las secuencias de coordenadas a lo largo del
tiempo y aprende patrones que corresponden a diferentes niveles
de ejecución técnica.

-------------------------------------------------------------
Estructura general del módulo:
1. Carga de datos JSON
2. Normalización y preprocesamiento
3. Definición y entrenamiento del modelo LSTM
4. Evaluación y visualización de métricas
5. Predicción de nuevas secuencias
-------------------------------------------------------------
"""

import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping

# ============================================================
# 1. CARGA DE ARCHIVOS JSON
# ============================================================

def load_all_jsons(base_dir):
    """
    Recorre recursivamente la carpeta base y carga todos los
    archivos JSON con coordenadas.

    Parámetros
    ----------
    base_dir : str
        Ruta a la carpeta raíz que contiene subcarpetas con
        archivos JSON de coordenadas.

    Retorna
    -------
    sequences : list[np.ndarray]
        Lista con las secuencias de coordenadas concatenadas
        (Pelvis, CodoDerechoReferencia, ManoDerechaReferencia).
        Cada elemento tiene forma (n_frames, 6).
    labels : list[int]
        Lista con las calificaciones (grades) asociadas.
    """
    sequences = []
    labels = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Cargar contenido del archivo JSON
                with open(file_path, "r") as f:
                    data = json.load(f)

                meta = data.get("metadata", {})
                grade = int(meta.get("grade", 0))  # Calificación 1–10

                # Extraer las coordenadas relevantes
                pelvis = np.array(data.get("Pelvis", []))
                elbow = np.array(data.get("Codo Derecha Referencia", []))
                hand = np.array(data.get("Mano Derecha Referencia", []))

                # Asegurar misma longitud para todas las secuencias
                min_len = min(len(pelvis), len(elbow), len(hand))
                pelvis, elbow, hand = pelvis[:min_len], elbow[:min_len], hand[:min_len]

                # Concatenar (x, y) de cada punto → shape (frames, 6)
                sequence = np.concatenate([pelvis, elbow, hand], axis=1)

                sequences.append(sequence)
                labels.append(grade)

    print(f"[INFO] Se cargaron {len(sequences)} clips desde {base_dir}")
    return sequences, labels


# ============================================================
# 2. PREPROCESAMIENTO DE DATOS
# ============================================================

def prepare_data(sequences, labels, max_len=100):
    """
    Normaliza, rellena y divide los datos en conjuntos de
    entrenamiento y prueba.

    Pasos:
    1. Normaliza cada clip (z-score)
    2. Rellena las secuencias hasta `max_len`
    3. Divide en train/test (80/20)

    Parámetros
    ----------
    sequences : list[np.ndarray]
        Lista de secuencias (n_frames, 6)
    labels : list[int]
        Lista con calificaciones (1–10)
    max_len : int
        Longitud máxima de secuencia (rellena con ceros)

    Retorna
    -------
    X_train, X_test : np.ndarray
        Tensores de entrada para LSTM (n_samples, max_len, 6)
    y_train, y_test : np.ndarray
        Etiquetas de entrenamiento y prueba
    """
    normalized = []

    # Normalizar coordenadas dentro de cada clip
    for seq in sequences:
        seq = (seq - np.mean(seq, axis=0)) / (np.std(seq, axis=0) + 1e-8)
        normalized.append(seq)

    # Igualar longitudes (relleno o truncamiento)
    X = pad_sequences(normalized, maxlen=max_len, dtype='float32',
                      padding='post', truncating='post')
    y = np.array(labels, dtype=np.int32)

    # Convertir etiquetas 1–10 a 0–9 (para softmax)
    y -= 1

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Datos preparados: {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


# ============================================================
# 3. DEFINICIÓN DEL MODELO LSTM
# ============================================================

def create_lstm_model(input_shape, num_classes=10):
    """
    Define la arquitectura del modelo LSTM.

    Arquitectura:
        LSTM(128, return_sequences=True)
        Dropout(0.3)
        LSTM(64)
        Dropout(0.3)
        Dense(64, relu)
        Dense(num_classes, softmax)

    Parámetros
    ----------
    input_shape : tuple
        (max_len, num_features)
    num_classes : int
        Número de clases de salida (10 calificaciones)

    Retorna
    -------
    model : keras.Model
        Modelo LSTM compilado y listo para entrenamiento.
    """
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compilación con optimizador y métrica
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ============================================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================================

def train_model(args):
    """
    Entrena un modelo LSTM a partir de todos los JSON de coordenadas.

    Parámetros
    ----------
    args.directory : str
        Ruta a la carpeta "Coordinates" con los JSONs.
    args.model_path : str
        Ruta donde se guardará el modelo (.h5).
    """
    data_dir = args.directory
    model_path = args.model_path

    # --- Cargar y preparar datos ---
    sequences, labels = load_all_jsons(data_dir)
    X_train, X_test, y_train, y_test = prepare_data(sequences, labels)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # --- Crear modelo ---
    model = create_lstm_model(input_shape)
    model.summary()

    # --- Callbacks para detener entrenamiento temprano ---
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # --- Entrenar modelo ---
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # --- Guardar modelo entrenado ---
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[INFO] Modelo guardado en {model_path}")

    # --- Evaluar desempeño ---
    evaluate_model(model, X_test, y_test)


# ============================================================
# 5. EVALUACIÓN DEL MODELO
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo LSTM y muestra las métricas de desempeño.

    Métricas mostradas:
        - Accuracy global
        - Reporte de clasificación (precision, recall, F1)
        - Matriz de confusión visual

    Parámetros
    ----------
    model : keras.Model
        Modelo entrenado.
    X_test : np.ndarray
        Datos de prueba.
    y_test : np.ndarray
        Etiquetas reales de prueba.
    """
    # --- Predicciones ---
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # --- Accuracy global ---
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}")

    # --- Reporte de clasificación detallado ---
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, digits=3))

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(1, 11), yticklabels=range(1, 11))
    plt.title("Matriz de Confusión (Grades 1–10)")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.show()


# ============================================================
# 6. PREDICCIÓN DE NUEVOS CLIPS
# ============================================================

def predict_clip(args):
    """
    Usa un modelo entrenado para estimar la calificación (1–10)
    de un nuevo clip JSON.

    Parámetros
    ----------
    args.file : str
        Ruta al archivo JSON del clip.
    args.model_path : str
        Ruta al modelo LSTM (.h5) entrenado.

    Retorna
    -------
    prediction : int
        Calificación estimada (1–10)
    """
    model_path = args.model_path
    json_file = args.file

    # --- Cargar modelo ---
    model = load_model(model_path)
    print(f"[INFO] Modelo cargado desde {model_path}")

    # --- Cargar y preparar clip ---
    with open(json_file, "r") as f:
        data = json.load(f)

    pelvis = np.array(data.get("Pelvis", []))
    elbow = np.array(data.get("Codo Derecha Referencia", []))
    hand = np.array(data.get("Mano Derecha Referencia", []))

    # Alinear longitudes
    min_len = min(len(pelvis), len(elbow), len(hand))
    sequence = np.concatenate([pelvis[:min_len], elbow[:min_len], hand[:min_len]], axis=1)

    # Normalización
    sequence = (sequence - np.mean(sequence, axis=0)) / (np.std(sequence, axis=0) + 1e-8)
    sequence = np.expand_dims(sequence, axis=0)  # (1, n_frames, 6)

    # --- Predicción ---
    prediction = np.argmax(model.predict(sequence), axis=1)[0] + 1  # volver a escala 1–10
    print(f"\n🎯 Calificación estimada: {prediction}/10\n")

    return prediction
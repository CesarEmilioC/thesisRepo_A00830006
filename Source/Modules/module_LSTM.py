"""
-------------------------------------------------------------
Module: module_LSTM.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / LSTM for Paddle Tennis Analysis
-------------------------------------------------------------
Descripción general:
--------------------
Este módulo implementa todas las funciones necesarias para:

1. Cargar y procesar coordenadas JSON (pose estimation)
2. Entrenar un modelo LSTM mejorado (Bidirectional + BatchNorm)
3. Evaluar el modelo con métricas y visualizaciones
4. Realizar predicciones sobre clips individuales
5. Analizar la cantidad de clips por calificación (countGrades)
6. Visualizar la matriz de confusión y un gráfico de barras por clase

El objetivo es construir un pipeline robusto para tu tesis.
-------------------------------------------------------------
"""

# ============================================================
# IMPORTACIONES
# ============================================================

import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

from keras.models import Sequential, load_model
from keras.layers import (
    LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional
)
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping


# ============================================================
# 1. CARGA DE ARCHIVOS JSON
# ============================================================

def load_all_jsons(base_dir):
    """
    Recorre recursivamente la carpeta base y carga todos los JSON
    con coordenadas de Pelvis, Codo Derecha y Mano Derecha.

    Cada JSON debe tener metadata.grade indicando la calificación.

    Retorna:
    --------
    sequences : list(np.ndarray) con forma (frames, 6)
    labels    : list(int) calificación del clip
    """

    sequences = []
    labels = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)

                with open(path, "r") as f:
                    data = json.load(f)

                meta = data.get("metadata", {})
                grade = int(meta.get("grade", 0))

                pelvis = np.array(data.get("Pelvis", []))
                elbow  = np.array(data.get("Codo Derecha Referencia", []))
                hand   = np.array(data.get("Mano Derecha Referencia", []))

                # cortar a mínimo común
                L = min(len(pelvis), len(elbow), len(hand))
                if L == 0:
                    continue

                seq = np.concatenate([
                    pelvis[:L], elbow[:L], hand[:L]
                ], axis=1)  # (frames, 6)

                sequences.append(seq)
                labels.append(grade)

    print(f"[INFO] Se cargaron {len(sequences)} clips desde {base_dir}")
    return sequences, labels


# ============================================================
# 2. CONTAR GRADES (Nuevo subcomando)
# ============================================================

def count_grades(args):
    """
    Cuenta cuántos clips existen por calificación (grade).
    Llamado desde main:
        python main.py countGrades --directory Coordinates
    """

    _, labels = load_all_jsons(args.directory)

    if not labels:
        print("[ERROR] No hay JSONs con calificaciones.")
        return

    counter = Counter(labels)

    print("\n============================")
    print(" DISTRIBUCIÓN DE CALIFICACIONES")
    print("============================\n")

    for g in sorted(counter.keys()):
        print(f"Grade {g}: {counter[g]} clips")

    print("\n============================\n")


# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================

def prepare_data(sequences, labels, max_len=120):
    """
    Normaliza secuencias (z-score), aplica padding / truncamiento
    y crea el split train-test.

    Retorna:
    --------
    X_train, X_test : arrays (N, max_len, 6)
    y_train, y_test : arrays (N,)
    """

    normalized = [(seq - np.mean(seq, axis=0)) /
                  (np.std(seq, axis=0) + 1e-8)
                  for seq in sequences]

    X = pad_sequences(
        normalized,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )

    y = np.array(labels) - 1  # convertir 1–10 → 0–9

    counts = Counter(y)

    # algunas clases tienen <2 muestras → no se puede estratificar
    stratify_val = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42,
        stratify=stratify_val
    )

    print(f"[INFO] Train: {len(X_train)} / Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. MODELO LSTM MEJORADO
# ============================================================

def create_lstm_model(input_shape, num_classes=10):
    """
    Modelo LSTM robusto para capturar dinámica temporal de movimiento.
    Incluye:
    - BatchNormalization
    - Bidirectional LSTM profundo
    - Dropout
    """

    model = Sequential([
        BatchNormalization(),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.30),

        Bidirectional(LSTM(64)),
        Dropout(0.25),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. ENTRENAMIENTO DEL MODELO
# ============================================================

def train_model(args):
    """
    Entrena el modelo LSTM sobre la carpeta Coordinates/
    """

    seqs, labels = load_all_jsons(args.directory)
    X_train, X_test, y_train, y_test = prepare_data(seqs, labels)

    model = create_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    early = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early],
        verbose=1
    )

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)

    print(f"[INFO] Modelo guardado en {args.model_path}")

    evaluate_model(model, X_test, y_test)


# ============================================================
# 6. VISUALIZACIONES Y MÉTRICAS
# ============================================================

def plot_class_distribution(y_true, y_pred):
    """
    Dibuja un gráfico de barras comparando:
    - cantidad real por clase
    - cantidad predicha por clase
    """

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    labels = sorted(set(list(true_counts.keys()) + list(pred_counts.keys())))

    real = [true_counts.get(l, 0) for l in labels]
    pred = [pred_counts.get(l, 0) for l in labels]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))

    plt.bar(x - 0.2, real, width=0.4, label="Real")
    plt.bar(x + 0.2, pred, width=0.4, label="Predicho")

    plt.xticks(x, [l + 1 for l in labels])
    plt.title("Distribución por clase (Real vs Predicho)")
    plt.xlabel("Clase (Grade)")
    plt.ylabel("Cantidad de clips")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y genera:
    - Accuracy
    - Classification report (una sola vez)
    - Matriz de confusión
    - Gráfico de barras (Real vs Predicho)
    """

    # predicciones (silenciado)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")

    # clasificación sin duplicados
    valid_labels = sorted(set(y_test))

    print("📊 Reporte de Clasificación:\n")
    print(classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0,
        labels=valid_labels
    ))

    # matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=valid_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=[v + 1 for v in valid_labels],
        yticklabels=[v + 1 for v in valid_labels]
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión (Grades)")
    plt.show()

    # gráfico de barras por clase
    plot_class_distribution(y_test, y_pred)


# ============================================================
# 7. PREDICCIÓN DE NUEVOS CLIPS
# ============================================================

def predict_clip(args):
    """
    Predice la calificación de un clip JSON individual.
    """

    model = load_model(args.model_path)
    print(f"[INFO] Modelo cargado desde {args.model_path}")

    with open(args.file, "r") as f:
        data = json.load(f)

    pelvis = np.array(data.get("Pelvis", []))
    elbow  = np.array(data.get("Codo Derecha Referencia", []))
    hand   = np.array(data.get("Mano Derecha Referencia", []))

    L = min(len(pelvis), len(elbow), len(hand))
    sequence = np.concatenate(
        [pelvis[:L], elbow[:L], hand[:L]], axis=1
    )

    sequence = (sequence - np.mean(sequence, axis=0)) / (np.std(sequence, axis=0) + 1e-8)
    sequence = np.expand_dims(sequence, axis=0)

    pred = np.argmax(model.predict(sequence, verbose=0), axis=1)[0] + 1

    print(f"\n🎯 Calificación estimada: {pred}/10\n")
    return pred
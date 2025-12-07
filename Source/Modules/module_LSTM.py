"""
-------------------------------------------------------------
Module: module_LSTM.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / LSTM for Paddle Tennis Analysis
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
from datetime import datetime

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

                L = min(len(pelvis), len(elbow), len(hand))
                if L == 0:
                    continue

                seq = np.concatenate([pelvis[:L], elbow[:L], hand[:L]], axis=1)

                sequences.append(seq)
                labels.append(grade)

    print(f"[INFO] Se cargaron {len(sequences)} clips desde {base_dir}")
    return sequences, labels


# ============================================================
# 2. CONTADOR DE GRADES
# ============================================================

def count_grades(args):
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

def prepare_data(sequences, labels, max_len=90):
    normalized = [
        (seq - np.mean(seq, axis=0)) / (np.std(seq, axis=0) + 1e-8)
        for seq in sequences
    ]

    X = pad_sequences(
        normalized,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )

    y = np.array(labels) - 1  

    counts = Counter(y)
    stratify_val = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_val
    )

    print(f"[INFO] Train: {len(X_train)} / Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. MODELO LSTM
# ============================================================

def create_lstm_model(input_shape, num_classes=10):
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
# 5. ENTRENAMIENTO + GUARDADO DE RESULTADOS
# ============================================================

def train_model(args):
    """
    Entrena el modelo LSTM y guarda:
    - history.json
    - learning_curves.png
    - confusion_matrix.png
    - class_distribution.png
    - classification_report.txt
    - modelo .h5
    """

    # --------------------------------------------------------
    # CREAR CARPETA DE RESULTADOS
    # --------------------------------------------------------
    # Fecha en formato dd-mm-yy
    date_str = datetime.now().strftime("%d-%m-%y")
    results_dir = os.path.join("Results", f"{args.run_name}_{date_str}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Guardando resultados en: {results_dir}")

    # --------------------------------------------------------
    # CARGA + PREPROCESAMIENTO
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # ENTRENAR
    # --------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early],
        verbose=1
    )

    # --------------------------------------------------------
    # GUARDAR HISTORIAL
    # --------------------------------------------------------
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=4)
    print(f"[INFO] Historial guardado en {history_path}")

    # --------------------------------------------------------
    # GUARDAR CURVAS DE APRENDIZAJE
    # --------------------------------------------------------
    learning_curve_path = os.path.join(results_dir, "learning_curves.png")

    plt.figure(figsize=(10, 6))

    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Learning Curves - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Learning Curves - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(learning_curve_path, dpi=300)
    plt.close()
    print(f"[INFO] Curvas guardadas en {learning_curve_path}")

    # --------------------------------------------------------
    # GUARDAR MODELO
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    print(f"[INFO] Modelo guardado en {args.model_path}")

    # --------------------------------------------------------
    # EVALUACIÓN FINAL
    # --------------------------------------------------------
    evaluate_model(model, X_test, y_test, results_dir)


# ============================================================
# 6. MÉTRICAS + FIGURAS
# ============================================================

def plot_class_distribution(y_true, y_pred, save_path=None):
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
    plt.title("Real vs Predicho")
    plt.xlabel("Clase (Grade)")
    plt.ylabel("Clips")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Gráfico de distribución guardado en {save_path}")

    plt.show()


def evaluate_model(model, X_test, y_test, results_dir=None):

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")

    valid_labels = sorted(set(y_test))
    report = classification_report(
        y_test, y_pred, digits=3,
        zero_division=0, labels=valid_labels
    )

    print("📊 Reporte de Clasificación:\n")
    print(report)

    # Guardar reporte
    if results_dir:
        report_path = os.path.join(results_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"[INFO] Reporte guardado en {report_path}")

    # ------------------------------------------
    # Matriz de confusión
    # ------------------------------------------
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
    plt.title("Matriz de Confusión")
    plt.tight_layout()

    if results_dir:
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        print(f"[INFO] Matriz guardada en {cm_path}")

    plt.show()

    # ------------------------------------------
    # Distribución por clase
    # ------------------------------------------
    if results_dir:
        bar_path = os.path.join(results_dir, "class_distribution.png")
        plot_class_distribution(y_test, y_pred, save_path=bar_path)
    else:
        plot_class_distribution(y_test, y_pred)


# ============================================================
# 7. PREDICCIÓN DE CLIPS
# ============================================================

def predict_clip(args):
    model = load_model(args.model_path)
    print(f"[INFO] Modelo cargado desde {args.model_path}")

    with open(args.file, "r") as f:
        data = json.load(f)

    pelvis = np.array(data.get("Pelvis", []))
    elbow  = np.array(data.get("Codo Derecha Referencia", []))
    hand   = np.array(data.get("Mano Derecha Referencia", []))

    L = min(len(pelvis), len(elbow), len(hand))
    sequence = np.concatenate([pelvis[:L], elbow[:L], hand[:L]], axis=1)

    sequence = (sequence - np.mean(sequence, axis=0)) / (np.std(sequence, axis=0) + 1e-8)
    sequence = np.expand_dims(sequence, axis=0)

    pred = np.argmax(model.predict(sequence, verbose=0), axis=1)[0] + 1

    print(f"\n🎯 Calificación estimada: {pred}/10\n")
    return pred
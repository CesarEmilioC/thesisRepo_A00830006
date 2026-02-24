"""
-------------------------------------------------------------
Module: module_LSTM.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / LSTM for Paddle Tennis Analysis
-------------------------------------------------------------
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import argparse
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from collections import Counter
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

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

import config


# ============================================================
# 1. JSON FILE LOADING
# ============================================================

def load_all_jsons(base_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load all JSON coordinate files recursively from base_dir.

    Returns a tuple of (sequences, labels) where each sequence is a numpy
    array of shape (num_frames, 8) containing pelvis + shoulder + elbow + wrist
    relative coordinates (2D each = 8 features), and each label is the grade (1-10).

    If a JSON does not contain the shoulder field (older 3-keypoint format),
    it is still loaded with 6 features for backward compatibility, but a
    warning is printed.
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

                pelvis   = np.array(data.get(config.FIELD_PELVIS, []))
                shoulder = np.array(data.get(config.FIELD_SHOULDER_RELATIVE, []))
                elbow    = np.array(data.get(config.FIELD_ELBOW_RELATIVE, []))
                hand     = np.array(data.get(config.FIELD_WRIST_RELATIVE, []))

                has_shoulder = len(shoulder) > 0

                if has_shoulder:
                    L = min(len(pelvis), len(shoulder), len(elbow), len(hand))
                else:
                    L = min(len(pelvis), len(elbow), len(hand))

                if L == 0:
                    print(f"[WARNING] Skipping empty sequence: {path}")
                    continue

                if has_shoulder:
                    seq = np.concatenate(
                        [pelvis[:L], shoulder[:L], elbow[:L], hand[:L]], axis=1
                    )
                else:
                    print(f"[WARNING] No shoulder data in {file}, loading with 6 features")
                    seq = np.concatenate(
                        [pelvis[:L], elbow[:L], hand[:L]], axis=1
                    )

                sequences.append(seq)
                labels.append(grade)

    print(f"[INFO] Loaded {len(sequences)} clips from {base_dir}")
    return sequences, labels


# ============================================================
# 2. GRADE COUNTER
# ============================================================

def count_grades(args: argparse.Namespace) -> None:
    _, labels = load_all_jsons(args.directory)

    if not labels:
        print("[ERROR] No JSONs with grades found.")
        return

    grade_counter = Counter(labels)

    print("\n============================")
    print(" GRADE DISTRIBUTION (1-10)")
    print("============================\n")

    for g in sorted(grade_counter.keys()):
        print(f"Grade {g}: {grade_counter[g]} clips")

    # Also show 5-class distribution
    class_labels_list = [config.grade_to_class(g) for g in labels]
    class_counter = Counter(class_labels_list)

    print("\n============================")
    print(" CLASS DISTRIBUTION (5 classes)")
    print("============================\n")

    for c in sorted(class_counter.keys()):
        print(f"Class {c} ({config.CLASS_LABELS[c]}): {class_counter[c]} clips")

    print(f"\nTotal clips: {len(labels)}")
    print("============================\n")


# ============================================================
# 3. PREPROCESSING
# ============================================================

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize a single sequence using per-feature zero-mean, unit-variance normalization.

    Each clip is normalized independently (per-clip normalization).
    The same function is used during both training and inference.

    Parameters
    ----------
    sequence : np.ndarray
        Shape (timesteps, features). Raw coordinate sequence.

    Returns
    -------
    np.ndarray
        Normalized sequence with the same shape.
    """
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0) + config.NORMALIZATION_EPSILON
    return (sequence - mean) / std


def prepare_data(sequences: List[np.ndarray], labels: List[int],
                  max_len: int = config.MAX_SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize, pad, and split sequences into train/test sets."""
    normalized = [normalize_sequence(seq) for seq in sequences]

    X = pad_sequences(
        normalized,
        maxlen=max_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )

    # Grade labels are 1-10 in the data. Map to 5 classes using config.grade_to_class():
    #   Grade 1-2 -> Class 0 (Very Low)
    #   Grade 3-4 -> Class 1 (Low)
    #   Grade 5-6 -> Class 2 (Medium)
    #   Grade 7-8 -> Class 3 (High)
    #   Grade 9-10 -> Class 4 (Excellent)
    y = np.array([config.grade_to_class(g) for g in labels])

    counts = Counter(y)
    stratify_val = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_val
    )

    print(f"[INFO] Train: {len(X_train)} / Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. LSTM MODEL
# ============================================================

def create_lstm_model(input_shape: Tuple[int, int], num_classes: int = config.NUM_CLASSES) -> Sequential:
    model = Sequential([
        BatchNormalization(),

        Bidirectional(LSTM(config.LSTM_UNITS_L1, return_sequences=True)),
        Dropout(config.DROPOUT_L1),

        Bidirectional(LSTM(config.LSTM_UNITS_L2)),
        Dropout(config.DROPOUT_L2),

        Dense(config.DENSE_UNITS, activation='relu'),
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
# 5. TRAINING + SAVING RESULTS
# ============================================================

def train_model(args: argparse.Namespace) -> None:
    """Train the LSTM model and save results (history, plots, metrics, model)."""

    if not os.path.isdir(args.directory):
        print(f"[ERROR] Directory not found: {args.directory}")
        return

    # --------------------------------------------------------
    # CREATE RESULTS FOLDER
    # --------------------------------------------------------
    date_str = datetime.now().strftime("%d-%m-%y")
    results_dir = os.path.join(config.RESULTS_DIR, f"{args.run_name}_{date_str}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {results_dir}")

    # --------------------------------------------------------
    # LOAD + PREPROCESS
    # --------------------------------------------------------
    seqs, labels = load_all_jsons(args.directory)

    if len(seqs) == 0:
        print("[ERROR] No valid sequences loaded. Check the directory path and JSON format.")
        return

    unique_grades = sorted(set(labels))
    print(f"[INFO] Grades found: {unique_grades} ({len(unique_grades)} unique grades)")
    unique_classes = sorted(set(config.grade_to_class(g) for g in labels))
    class_names = [config.CLASS_LABELS[c] for c in unique_classes]
    print(f"[INFO] Classes mapped: {list(zip(unique_classes, class_names))} ({len(unique_classes)} of {config.NUM_CLASSES} classes)")

    X_train, X_test, y_train, y_test = prepare_data(seqs, labels)

    model = create_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    early = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    # Compute class weights to handle class imbalance
    unique_train_classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=unique_train_classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(unique_train_classes, cw)}
    print(f"[INFO] Class weights: {class_weight_dict}")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early],
        class_weight=class_weight_dict,
        verbose=1
    )

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=4)
    print(f"[INFO] History saved to {history_path}")

    # --------------------------------------------------------
    # SAVE LEARNING CURVES
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
    print(f"[INFO] Learning curves saved to {learning_curve_path}")

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    print(f"[INFO] Model saved to {args.model_path}")

    # Also save a copy of the model in the results directory for reproducibility
    results_model_path = os.path.join(results_dir, "lstm_model.h5")
    model.save(results_model_path)
    print(f"[INFO] Model copy saved to {results_model_path}")

    # --------------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------------
    evaluate_model(model, X_test, y_test, results_dir)


# ============================================================
# 6. METRICS + PLOTS
# ============================================================

def plot_class_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None) -> None:
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    all_classes = sorted(set(list(true_counts.keys()) + list(pred_counts.keys())))
    real = [true_counts.get(c, 0) for c in all_classes]
    pred = [pred_counts.get(c, 0) for c in all_classes]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(all_classes))

    plt.bar(x - 0.2, real, width=0.4, label="True")
    plt.bar(x + 0.2, pred, width=0.4, label="Predicted")

    tick_labels = [config.CLASS_LABELS[c] if c < len(config.CLASS_LABELS) else str(c) for c in all_classes]
    plt.xticks(x, tick_labels, rotation=15)
    plt.title("True vs Predicted Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Clips")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[INFO] Distribution plot saved to {save_path}")
    else:
        plt.show()


def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray,
                    results_dir: Optional[str] = None) -> None:

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc:.4f}\n")

    valid_labels = sorted(set(y_test))
    target_names = [config.CLASS_LABELS[c] if c < len(config.CLASS_LABELS) else str(c) for c in valid_labels]
    report = classification_report(
        y_test, y_pred, digits=3,
        zero_division=0, labels=valid_labels,
        target_names=target_names
    )

    print("[INFO] Classification Report:\n")
    print(report)

    if results_dir:
        report_path = os.path.join(results_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(report)
        print(f"[INFO] Report saved to {report_path}")

    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------
    cm_data = confusion_matrix(y_test, y_pred, labels=valid_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_data,
        annot=True,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if results_dir:
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {cm_path}")
    else:
        plt.show()

    # ------------------------------------------
    # Class Distribution
    # ------------------------------------------
    if results_dir:
        bar_path = os.path.join(results_dir, "class_distribution.png")
        plot_class_distribution(y_test, y_pred, save_path=bar_path)
    else:
        plot_class_distribution(y_test, y_pred)


# ============================================================
# 7. CLIP PREDICTION
# ============================================================

def predict_clip(args: argparse.Namespace) -> int:
    """Predict the quality grade of a single clip using a trained LSTM model."""

    if not os.path.isfile(args.file):
        print(f"[ERROR] JSON file not found: {args.file}")
        return -1
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        return -1

    model = load_model(args.model_path)
    print(f"[INFO] Model loaded from {args.model_path}")

    with open(args.file, "r") as f:
        data = json.load(f)

    pelvis   = np.array(data.get(config.FIELD_PELVIS, []))
    shoulder = np.array(data.get(config.FIELD_SHOULDER_RELATIVE, []))
    elbow    = np.array(data.get(config.FIELD_ELBOW_RELATIVE, []))
    hand     = np.array(data.get(config.FIELD_WRIST_RELATIVE, []))

    has_shoulder = len(shoulder) > 0
    if has_shoulder:
        L = min(len(pelvis), len(shoulder), len(elbow), len(hand))
        sequence = np.concatenate(
            [pelvis[:L], shoulder[:L], elbow[:L], hand[:L]], axis=1
        )
    else:
        L = min(len(pelvis), len(elbow), len(hand))
        sequence = np.concatenate([pelvis[:L], elbow[:L], hand[:L]], axis=1)

    sequence = normalize_sequence(sequence)

    # Pad/truncate to match training input shape (MAX_SEQUENCE_LENGTH frames)
    sequence = pad_sequences(
        [sequence],
        maxlen=config.MAX_SEQUENCE_LENGTH,
        dtype='float32',
        padding='post',
        truncating='post'
    )
    # sequence shape is now (1, MAX_SEQUENCE_LENGTH, features)

    # Get predicted class (0-4)
    pred_class = np.argmax(model.predict(sequence, verbose=0), axis=1)[0]
    class_label = config.CLASS_LABELS[pred_class] if pred_class < len(config.CLASS_LABELS) else str(pred_class)

    print(f"\n[RESULT] Predicted class: {pred_class} ({class_label})\n")
    return pred_class
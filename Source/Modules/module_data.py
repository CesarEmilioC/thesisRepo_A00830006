"""
-------------------------------------------------------------
Module: module_data.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / Paddle Tennis Movement Feedback System
-------------------------------------------------------------
Shared data-handling, preprocessing, evaluation, and analysis utilities
used by all model modules (LSTM, GRU, TCN) and thesis artifact generation.

Functions:
    Data loading:
        load_all_jsons          Load JSON coordinate files recursively.
    Preprocessing:
        normalize_sequence      Per-clip zero-mean/unit-variance normalization.
        augment_sequences       Gaussian noise augmentation.
        prepare_data            Full pipeline: normalize, pad, split, augment.
    Evaluation & result saving:
        evaluate_model          Confusion matrix, classification report, plots.
        plot_class_distribution True vs predicted class bar chart.
        save_training_history   Save Keras history dict to JSON.
        save_learning_curves    Save loss/accuracy plots.
        get_next_test_number    Auto-detect next experiment number.
    Dataset analysis:
        count_grades            Print grade and class distributions.
        build_test_set          Reconstruct canonical test split for inference.
        compute_spearman        Spearman rank correlation for final models.
        dataset_stats           Per-player/class/grade clip counts.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import argparse
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import pad_sequences

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
    """Print grade (1-10) and class (5-class) distributions."""
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
    """
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0) + config.NORMALIZATION_EPSILON
    return (sequence - mean) / std


def augment_sequences(sequences: List[np.ndarray], labels: List[int],
                      factor: int = config.AUGMENTATION_FACTOR,
                      noise_std: float = config.AUGMENTATION_NOISE_STD
                      ) -> Tuple[List[np.ndarray], List[int]]:
    """Create augmented copies of sequences by adding Gaussian noise."""
    aug_seqs = list(sequences)
    aug_labels = list(labels)

    for _ in range(factor):
        for seq, lbl in zip(sequences, labels):
            noisy = seq + np.random.normal(0, noise_std, seq.shape)
            aug_seqs.append(noisy)
            aug_labels.append(lbl)

    print(f"[INFO] Augmented {len(sequences)} -> {len(aug_seqs)} sequences (factor={factor})")
    return aug_seqs, aug_labels


def prepare_data(sequences: List[np.ndarray], labels: List[int],
                  max_len: int = config.MAX_SEQUENCE_LENGTH,
                  augment: bool = True,
                  test_size: float = 0.2,
                  random_state: int = 42
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize, pad, and split sequences into train/test sets.

    Grade labels (1-10) are mapped to 5 classes via config.grade_to_class().
    Augmentation is applied to the training set only (after splitting) to
    prevent data leakage. ``test_size`` and ``random_state`` are exposed so
    the multi-split sensitivity analysis can vary the train fraction while
    keeping the same canonical seed.
    """
    y_all = np.array([config.grade_to_class(g) for g in labels])

    counts = Counter(y_all)
    stratify_val = y_all if min(counts.values()) >= 2 else None

    seq_train, seq_test, y_train, y_test = train_test_split(
        sequences, y_all, test_size=test_size,
        random_state=random_state, stratify=stratify_val
    )

    if augment:
        seq_train_list = list(seq_train)
        y_train_list = list(y_train)
        seq_train_list, y_train_list = augment_sequences(seq_train_list, y_train_list)
        seq_train = seq_train_list
        y_train = np.array(y_train_list)

    norm_train = [normalize_sequence(s) for s in seq_train]
    norm_test = [normalize_sequence(s) for s in seq_test]

    X_train = pad_sequences(norm_train, maxlen=max_len, dtype='float32',
                            padding='post', truncating='post')
    X_test = pad_sequences(norm_test, maxlen=max_len, dtype='float32',
                           padding='post', truncating='post')

    print(f"[INFO] Train: {len(X_train)} / Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. RESULT-SAVING HELPERS
# ============================================================

def get_next_test_number(model_prefix: str) -> int:
    """Scan the Results directory and return the next test number for a given model.

    Looks for folders matching '{model_prefix}_Test{N}_{date}' and returns N+1.
    For the LSTM model, also checks legacy folders matching 'Test{N}_{date}'.
    """
    max_num = 0

    if not os.path.isdir(config.RESULTS_DIR):
        return 1

    for folder in os.listdir(config.RESULTS_DIR):
        match = re.match(rf'^{model_prefix}_Test(\d+)_', folder)
        if match:
            max_num = max(max_num, int(match.group(1)))
            continue

        if model_prefix == "LSTM":
            match = re.match(r'^Test(\d+)_', folder)
            if match:
                max_num = max(max_num, int(match.group(1)))

    return max_num + 1


def save_training_history(history, results_dir: str) -> None:
    """Save training history as JSON."""
    history_path = os.path.join(results_dir, "training_history.json")
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"[INFO] History saved to {history_path}")


def save_learning_curves(history, results_dir: str) -> None:
    """Save learning curves plot (loss + accuracy)."""
    learning_curve_path = os.path.join(results_dir, "learning_curves.png")

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Learning Curves - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

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


# ============================================================
# 5. METRICS + PLOTS
# ============================================================

def plot_class_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[str] = None) -> None:
    """Bar chart of true vs predicted class distribution."""
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


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                    results_dir: Optional[str] = None) -> None:
    """Evaluate a trained model: print metrics, save confusion matrix and class distribution."""

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

    if results_dir:
        bar_path = os.path.join(results_dir, "class_distribution.png")
        plot_class_distribution(y_test, y_pred, save_path=bar_path)
    else:
        plot_class_distribution(y_test, y_pred)


# ============================================================
# 6. THESIS ANALYSIS
# ============================================================

def build_test_set(seqs, labels, n_features):
    """Reconstruct the canonical test split (random_state=42).

    If *n_features* is 6, the current 8-feature sequences are projected by
    dropping shoulder columns (indices 2-3), preserving backward compatibility
    with LSTM Test01 which was trained without the shoulder keypoint.

    Returns (X_test, y_test) or (None, None) if no sequences match.
    """
    from tf_keras.utils import pad_sequences as kpad

    y_all = np.array([config.grade_to_class(g) for g in labels])
    counts = Counter(y_all)
    stratify = y_all if min(counts.values()) >= 2 else None

    _, seq_test, _, y_test = train_test_split(
        seqs, y_all, test_size=0.2, random_state=42, stratify=stratify
    )

    if n_features == 6:
        projected = []
        for s in seq_test:
            if s.shape[1] == 8:
                projected.append(np.concatenate([s[:, :2], s[:, 4:]], axis=1))
            elif s.shape[1] == 6:
                projected.append(s)
        filtered_seqs = projected
        filtered_y    = list(y_test)
    else:
        filtered_seqs = [s for s in seq_test if s.shape[1] == n_features]
        filtered_y    = [y_test[i] for i, s in enumerate(seq_test)
                         if s.shape[1] == n_features]

    if not filtered_seqs:
        return None, None

    norm = [normalize_sequence(s) for s in filtered_seqs]
    X = kpad(norm, maxlen=config.MAX_SEQUENCE_LENGTH, dtype='float32',
             padding='post', truncating='post')
    return X, np.array(filtered_y)


def compute_spearman(args: argparse.Namespace) -> None:
    """Compute Spearman rank correlation for the 3 final models.

    Reconstructs the test set using the canonical random_state=42 split and
    runs inference with each final model.  Output: spearman_results.txt saved
    to the thesis document folder.
    """
    from tf_keras.models import load_model as tf_load_model
    from tf_keras.utils import pad_sequences as kpad

    coords_dir = args.directory
    print(f"[INFO] Loading sequences from {coords_dir} ...")
    seqs, labels = load_all_jsons(coords_dir)

    y_all = np.array([config.grade_to_class(g) for g in labels])
    counts = Counter(y_all)
    stratify = y_all if min(counts.values()) >= 2 else None
    _, seq_test, _, y_test = train_test_split(
        seqs, y_all, test_size=0.2, random_state=42, stratify=stratify
    )
    norm = [normalize_sequence(s) for s in seq_test]
    X_test = kpad(norm, maxlen=config.MAX_SEQUENCE_LENGTH, dtype='float32',
                  padding='post', truncating='post')

    DISPLAY = {
        'LSTM_Test03': 'BiLSTM Final Model (Test03)',
        'GRU_Test01':  'BiGRU  Final Model (Test01)',
        'TCN_Test01':  'TCN    Final Model (Test01)',
    }

    lines = ["=== Spearman Rank Correlation "
             "(ordinal agreement, predicted vs true class) ===\n"]

    for exp in config.EXPERIMENTS:
        if exp['key'] not in config.FINAL_MODELS:
            continue
        model_path = exp['model_path']
        if not os.path.exists(model_path):
            print(f"  [SKIP] {exp['key']}: model not found")
            lines.append(f"{DISPLAY[exp['key']]}: MODEL NOT FOUND\n")
            continue

        model  = tf_load_model(config.safe_model_path(model_path))
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        rho, pval = spearmanr(y_test, y_pred)
        line = f"{DISPLAY[exp['key']]}: rho = {rho:.2f}, p = {pval:.4f}\n"
        lines.append(line)
        print(f"  [OK] {exp['key']}: rho={rho:.2f}, p={pval:.4f}")

    out_path = os.path.join(config.get_thesis_dir(args), 'spearman_results.txt')
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"[OK] spearman_results.txt -> {out_path}")


def dataset_stats(args: argparse.Namespace) -> None:
    """Compute and save dataset statistics to the thesis document folder.

    Counts clips per player, quality class, and raw grade.
    """
    coords_dir = args.directory
    print(f"[INFO] Scanning {coords_dir} ...")

    player_counter = Counter()
    class_counter  = Counter()
    grade_counter  = Counter()
    valid_ratios   = []
    interp_ratios  = []
    resolutions    = Counter()
    total          = 0

    for root, _, files in os.walk(coords_dir):
        for fname in files:
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(root, fname)) as f:
                data = json.load(f)

            total += 1
            meta  = data.get('metadata', {})
            grade = meta.get('grade')
            grade = int(grade) if grade is not None else 0

            m = re.search(r'player(\d+)', fname)
            if m:
                player_counter[int(m.group(1))] += 1

            if grade > 0:
                grade_counter[grade] += 1
                class_counter[config.grade_to_class(grade)] += 1

            pelvis = data.get(config.FIELD_PELVIS, [])
            if pelvis:
                valid = sum(1 for p in pelvis if any(v != 0 for v in p))
                valid_ratios.append(valid / len(pelvis))

            ir = meta.get('interpolated_frames_ratio')
            if ir is not None:
                interp_ratios.append(float(ir))

            res = meta.get('resolution') or meta.get('video_resolution')
            if res:
                resolutions[str(res)] += 1

    lines = [
        "=== Dataset Statistics ===\n",
        f"Total JSON files: {total}\n\n",
        "--- Clips per Player ---\n",
    ]
    for pid in sorted(player_counter):
        lines.append(f"  Player {pid}: {player_counter[pid]} clips\n")

    lines.append("\n--- Clips per Quality Class ---\n")
    for c in range(5):
        lines.append(f"  Class {c} ({config.CLASS_LABELS[c]}): {class_counter.get(c, 0)} clips\n")

    lines.append("\n--- Clips per Raw Grade (1-10) ---\n")
    for g in range(1, 11):
        lines.append(f"  Grade {g}: {grade_counter.get(g, 0)} clips\n")

    if valid_ratios:
        avg_v = np.mean(valid_ratios)
        lines += [f"\n--- Valid Frames ---\n",
                  f"  Average proportion per clip: {avg_v:.4f} ({avg_v*100:.1f}%)\n"]

    if interp_ratios:
        avg_i = np.mean(interp_ratios)
        lines += [f"\n--- Interpolated Frames ---\n",
                  f"  Average proportion per clip: {avg_i:.4f} ({avg_i*100:.1f}%)\n"]
    else:
        lines += [f"\n--- Interpolated Frames ---\n",
                  f"  Ratio not stored in JSON metadata.\n"]

    if resolutions:
        lines.append(f"\n--- Video Resolution (from metadata) ---\n")
        for res, cnt in resolutions.most_common():
            lines.append(f"  {res}: {cnt} clips\n")

    out_path = os.path.join(config.get_thesis_dir(args), 'dataset_stats.txt')
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"[OK] dataset_stats.txt -> {out_path}")
    for c in range(5):
        print(f"  {config.CLASS_LABELS[c]}: {class_counter.get(c, 0)}")

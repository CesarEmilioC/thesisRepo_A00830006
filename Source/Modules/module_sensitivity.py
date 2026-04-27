"""
-------------------------------------------------------------
Module: module_sensitivity.py
Author: Cesar Emilio Castano Marin
Project: Thesis - Smash Vision / Paddle Tennis Movement Feedback System
-------------------------------------------------------------
Sensitivity-analysis experiments for the Bidirectional LSTM model.

Two experiments are exposed via the CLI (``runSensitivity`` and
``runSplitAnalysis`` in ``main.py``):

    1. Loss x Optimizer (2 x 2 grid)
       A: sparse_categorical_crossentropy + Adam   (baseline)
       B: MSE + Adam
       C: sparse_categorical_crossentropy + SGD(momentum=0.9)
       D: MSE + SGD(momentum=0.9)

       For the MSE variants the architecture is held identical to the
       baseline (5-unit softmax output) and labels are one-hot encoded.
       Only the loss function differs across the four runs, so any
       performance gap is attributable solely to loss/optimizer choice.

    2. Multi-split analysis: train fractions [0.5, 0.6, 0.7, 0.8, 0.9],
       cross-entropy + Adam, same stratified split (random_state=42).

All other hyperparameters (epochs, batch_size, augmentation, sample
weights derived from balanced class weights, ReduceLROnPlateau, seed,
architecture) are kept identical to LSTM Test03.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import random
import argparse
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

from keras.models import Sequential
from keras.layers import (
    LSTM, Dense, Dropout,
    BatchNormalization, Bidirectional,
)
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from Modules.module_data import (
    load_all_jsons,
    prepare_data,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def _set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# MODEL FACTORY
# ============================================================

def _build_bilstm(input_shape, num_classes: int = config.NUM_CLASSES):
    """Bidirectional LSTM identical to ``module_LSTM.create_lstm_model``.

    Returned uncompiled so the caller can attach an arbitrary
    (loss, optimizer) pair from the sensitivity grid.
    """
    reg = l2(config.L2_REGULARIZATION)

    return Sequential([
        BatchNormalization(input_shape=input_shape),

        Bidirectional(LSTM(config.LSTM_UNITS_L1, return_sequences=True,
                           kernel_regularizer=reg, recurrent_regularizer=reg)),
        Dropout(config.DROPOUT_L1),

        Bidirectional(LSTM(config.LSTM_UNITS_L2,
                           kernel_regularizer=reg, recurrent_regularizer=reg)),
        Dropout(config.DROPOUT_L2),

        Dense(config.DENSE_UNITS, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),

        Dense(num_classes, activation='softmax'),
    ])


def _make_optimizer(name: str):
    if name == 'adam':
        return Adam()
    if name == 'sgd':
        return SGD(momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# PLOTTING HELPERS
# ============================================================

def _save_learning_curves(history, path: str) -> None:
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curves - Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curves - Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=config.PLOT_DPI)
    plt.close()


def _save_confusion_matrix(y_true, y_pred, valid_labels, target_names, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path, dpi=config.PLOT_DPI)
    plt.close()


def _save_class_distribution(y_true, y_pred, path: str) -> None:
    true_counts = Counter(int(v) for v in y_true)
    pred_counts = Counter(int(v) for v in y_pred)
    all_classes = sorted(set(list(true_counts.keys()) + list(pred_counts.keys())))
    real = [true_counts.get(c, 0) for c in all_classes]
    pred = [pred_counts.get(c, 0) for c in all_classes]

    x = np.arange(len(all_classes))
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, real, width=0.4, label='True')
    plt.bar(x + 0.2, pred, width=0.4, label='Predicted')
    tick_labels = [config.CLASS_LABELS[c] if c < len(config.CLASS_LABELS) else str(c)
                   for c in all_classes]
    plt.xticks(x, tick_labels, rotation=15)
    plt.title('True vs Predicted Class Distribution')
    plt.xlabel('Class'); plt.ylabel('Clips'); plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=config.PLOT_DPI)
    plt.close()


# ============================================================
# CORE TRAIN + EVALUATE
# ============================================================

def _train_and_evaluate(X_train, X_test, y_train, y_test,
                        loss_name: str, optimizer_name: str,
                        run_dir: str) -> dict:
    """Train one BiLSTM run and write all artifacts under ``run_dir``.

    Returns a metrics dict (also serialised to ``metrics.json``) plus the
    raw Keras history under the ``history`` key for downstream plotting.
    """
    _set_global_seed(SEED)

    os.makedirs(run_dir, exist_ok=True)

    model = _build_bilstm(input_shape=(X_train.shape[1], X_train.shape[2]))

    if loss_name == 'sparse_categorical_crossentropy':
        keras_loss = 'sparse_categorical_crossentropy'
        y_train_target = np.asarray(y_train)
    elif loss_name == 'mse':
        keras_loss = 'mse'
        y_train_target = to_categorical(y_train,
                                        num_classes=config.NUM_CLASSES).astype('float32')
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    model.compile(optimizer=_make_optimizer(optimizer_name),
                  loss=keras_loss,
                  metrics=['accuracy'])

    early = EarlyStopping(monitor='val_loss',
                          patience=config.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-6, verbose=1)

    unique_train_classes = np.unique(y_train)
    cw = compute_class_weight('balanced',
                              classes=unique_train_classes, y=y_train)
    class_weight_dict = {int(c): float(w)
                         for c, w in zip(unique_train_classes, cw)}
    sample_weight = np.array([class_weight_dict[int(c)] for c in y_train],
                             dtype='float32')

    print(f"[INFO] Class weights (balanced): {class_weight_dict}")
    print(f"[INFO] Loss = {loss_name}, Optimizer = {optimizer_name}, "
          f"n_train = {len(X_train)}, n_test = {len(X_test)}")

    history = model.fit(
        X_train, y_train_target,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early, reduce_lr],
        sample_weight=sample_weight,
        verbose=1,
    )

    # ---------- artifacts ----------
    history_path = os.path.join(run_dir, 'history.json')
    serial = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(serial, f, indent=4)

    _save_learning_curves(history, os.path.join(run_dir, 'learning_curves.png'))

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.asarray(y_test)

    acc         = float(accuracy_score(y_true, y_pred))
    macro_f1    = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    rho, pval = spearmanr(y_true, y_pred)
    rho  = float(rho)  if rho  is not None and not np.isnan(rho)  else float('nan')
    pval = float(pval) if pval is not None and not np.isnan(pval) else float('nan')

    valid_labels = sorted(set(int(v) for v in y_true))
    target_names = [config.CLASS_LABELS[c] if c < len(config.CLASS_LABELS) else str(c)
                    for c in valid_labels]
    report_dict = classification_report(
        y_true, y_pred, digits=3, zero_division=0,
        labels=valid_labels, target_names=target_names, output_dict=True,
    )
    with open(os.path.join(run_dir, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)

    _save_confusion_matrix(y_true, y_pred, valid_labels, target_names,
                           os.path.join(run_dir, 'confusion_matrix.png'))
    _save_class_distribution(y_true, y_pred,
                             os.path.join(run_dir, 'class_distribution.png'))

    metrics = {
        'loss': loss_name,
        'optimizer': optimizer_name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'spearman_rho': rho,
        'p_value': pval,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'epochs_run': int(len(history.history['loss'])),
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"[OK] run_dir = {run_dir}\n"
          f"     accuracy = {acc:.4f} | macro_f1 = {macro_f1:.4f} | "
          f"rho = {rho:.4f} (p = {pval:.4f})")

    return {**metrics, 'history': history.history}


# ============================================================
# EXPERIMENT 1 - LOSS x OPTIMIZER GRID
# ============================================================

GRID = [
    {'run_id': 'A_xent_adam',
     'loss': 'sparse_categorical_crossentropy', 'optimizer': 'adam',
     'label': 'CrossEntropy + Adam'},
    {'run_id': 'B_mse_adam',
     'loss': 'mse', 'optimizer': 'adam',
     'label': 'MSE + Adam'},
    {'run_id': 'C_xent_sgd',
     'loss': 'sparse_categorical_crossentropy', 'optimizer': 'sgd',
     'label': 'CrossEntropy + SGD(0.9)'},
    {'run_id': 'D_mse_sgd',
     'loss': 'mse', 'optimizer': 'sgd',
     'label': 'MSE + SGD(0.9)'},
]


def run_sensitivity(args: argparse.Namespace) -> None:
    """Loss x Optimizer 2 x 2 grid sensitivity analysis (BiLSTM only)."""
    if not os.path.isdir(args.directory):
        print(f"[ERROR] Directory not found: {args.directory}")
        return

    base_dir = os.path.join(config.RESULTS_DIR, 'SensitivityAnalysis')
    os.makedirs(base_dir, exist_ok=True)
    print(f"[INFO] Saving sensitivity-analysis results to: {base_dir}")

    seqs, labels = load_all_jsons(args.directory)
    if not seqs:
        print("[ERROR] No valid sequences loaded.")
        return

    # Same canonical 80/20 stratified split as the baseline (LSTM_Test03).
    _set_global_seed(SEED)
    X_train, X_test, y_train, y_test = prepare_data(
        seqs, labels, test_size=0.2, random_state=SEED, augment=True,
    )

    histories = {}
    summary_rows = []

    for entry in GRID:
        run_dir = os.path.join(base_dir, entry['run_id'])
        print(f"\n========== Sensitivity run: {entry['run_id']} "
              f"({entry['label']}) ==========")
        result = _train_and_evaluate(
            X_train, X_test, y_train, y_test,
            loss_name=entry['loss'], optimizer_name=entry['optimizer'],
            run_dir=run_dir,
        )
        histories[entry['run_id']] = result.pop('history')
        summary_rows.append({
            'run_id':       entry['run_id'],
            'label':        entry['label'],
            'loss':         entry['loss'],
            'optimizer':    entry['optimizer'],
            'accuracy':     result['accuracy'],
            'macro_f1':     result['macro_f1'],
            'spearman_rho': result['spearman_rho'],
            'p_value':      result['p_value'],
        })

    # ---- combined val_loss overlay ----
    overlay_path = os.path.join(base_dir, 'learning_curves_overlay.png')
    plt.figure(figsize=(10, 6))
    for entry in GRID:
        h = histories[entry['run_id']]
        plt.plot(h['val_loss'], label=entry['label'])
    plt.title('Validation Loss - Sensitivity Grid (Loss x Optimizer)')
    plt.xlabel('Epoch'); plt.ylabel('Validation Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(overlay_path, dpi=config.PLOT_DPI)
    plt.close()
    print(f"[INFO] learning_curves_overlay.png -> {overlay_path}")

    # ---- summary.csv ----
    csv_path = os.path.join(base_dir, 'summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('run_id,label,loss,optimizer,accuracy,macro_f1,spearman_rho,p_value\n')
        for r in summary_rows:
            f.write(f"{r['run_id']},{r['label']},{r['loss']},{r['optimizer']},"
                    f"{r['accuracy']:.4f},{r['macro_f1']:.4f},"
                    f"{r['spearman_rho']:.4f},{r['p_value']:.4f}\n")
    print(f"[INFO] summary.csv -> {csv_path}")


# ============================================================
# EXPERIMENT 2 - MULTI-SPLIT ANALYSIS
# ============================================================

TRAIN_FRACTIONS = [0.5, 0.6, 0.7, 0.8, 0.9]


def run_split_analysis(args: argparse.Namespace) -> None:
    """Multi-split sensitivity analysis (BiLSTM only).

    Repeats the BiLSTM training with train fractions in
    ``TRAIN_FRACTIONS``. Loss = sparse cross-entropy, optimizer = Adam,
    same architecture, augmentation, class weights, callbacks, and seed.
    """
    if not os.path.isdir(args.directory):
        print(f"[ERROR] Directory not found: {args.directory}")
        return

    base_dir = os.path.join(config.RESULTS_DIR, 'SplitAnalysis')
    os.makedirs(base_dir, exist_ok=True)
    print(f"[INFO] Saving split-analysis results to: {base_dir}")

    seqs, labels = load_all_jsons(args.directory)
    if not seqs:
        print("[ERROR] No valid sequences loaded.")
        return

    summary_rows = []

    for frac in TRAIN_FRACTIONS:
        run_dir = os.path.join(base_dir, f'{frac:.1f}')
        print(f"\n========== Split run: train_frac = {frac:.2f} ==========")

        _set_global_seed(SEED)
        X_train, X_test, y_train, y_test = prepare_data(
            seqs, labels, test_size=1.0 - frac, random_state=SEED, augment=True,
        )
        result = _train_and_evaluate(
            X_train, X_test, y_train, y_test,
            loss_name='sparse_categorical_crossentropy',
            optimizer_name='adam',
            run_dir=run_dir,
        )
        result.pop('history', None)
        summary_rows.append({
            'train_frac':   frac,
            'n_train':      result['n_train'],
            'n_test':       result['n_test'],
            'accuracy':     result['accuracy'],
            'macro_f1':     result['macro_f1'],
            'spearman_rho': result['spearman_rho'],
            'p_value':      result['p_value'],
        })

    # ---- summary.csv ----
    csv_path = os.path.join(base_dir, 'summary.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('train_frac,n_train,n_test,accuracy,macro_f1,spearman_rho,p_value\n')
        for r in summary_rows:
            f.write(f"{r['train_frac']:.2f},{r['n_train']},{r['n_test']},"
                    f"{r['accuracy']:.4f},{r['macro_f1']:.4f},"
                    f"{r['spearman_rho']:.4f},{r['p_value']:.4f}\n")
    print(f"[INFO] summary.csv -> {csv_path}")

    # ---- trend_plot.png ----
    fracs = [r['train_frac'] for r in summary_rows]
    accs  = [r['accuracy']   for r in summary_rows]
    f1s   = [r['macro_f1']   for r in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(fracs, accs, 'o-', label='Accuracy')
    plt.plot(fracs, f1s,  's-', label='Macro F1')
    plt.title('BiLSTM Performance vs Train Fraction')
    plt.xlabel('Train Fraction'); plt.ylabel('Score')
    plt.legend(); plt.grid(True); plt.tight_layout()
    trend_path = os.path.join(base_dir, 'trend_plot.png')
    plt.savefig(trend_path, dpi=config.PLOT_DPI)
    plt.close()
    print(f"[INFO] trend_plot.png -> {trend_path}")

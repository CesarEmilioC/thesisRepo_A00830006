"""
-------------------------------------------------------------
Module: module_TCN.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / TCN for Paddle Tennis Analysis
-------------------------------------------------------------
Temporal Convolutional Network (TCN) model used as a comparative
baseline against the LSTM. TCNs use dilated causal convolutions to
capture long-range temporal dependencies without recurrence, offering
parallelizable training and a fixed receptive field.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import argparse
import numpy as np
from datetime import datetime

from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')

from keras.models import Sequential, load_model
from keras.layers import (
    Conv1D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D,
    Activation, Add, Input
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from Modules.module_LSTM import (
    load_all_jsons,
    prepare_data,
    evaluate_model,
    normalize_sequence,
    save_training_history,
    save_learning_curves,
    get_next_test_number
)


# ============================================================
# TCN MODEL
# ============================================================

def create_tcn_model(input_shape, num_classes=config.NUM_CLASSES):
    """Build a TCN with stacked dilated causal convolutions.

    Architecture:
        Input -> BatchNorm -> [Conv1D(dilation=1) -> BN -> ReLU -> Dropout] x3
              -> GlobalAveragePooling -> Dense -> Dense(softmax)

    Dilated convolutions with rates [1, 2, 4] give an effective receptive
    field of 1 + 2*(3-1)*(1+2+4) = 29 frames per stack, sufficient for
    capturing the temporal patterns in 90-frame bandeja sequences.
    """
    reg = l2(config.L2_REGULARIZATION)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    for dilation_rate in config.TCN_DILATIONS:
        model.add(Conv1D(
            filters=config.TCN_FILTERS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=reg
        ))
        model.add(BatchNormalization())
        model.add(Dropout(config.TCN_DROPOUT))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(config.TCN_DENSE_UNITS, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(config.TCN_DROPOUT))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(args):
    if not os.path.isdir(args.directory):
        print(f"[ERROR] Directory not found: {args.directory}")
        return

    date_str = datetime.now().strftime("%d-%m-%y")
    test_num = get_next_test_number("TCN")
    run_name = f"Test{test_num:02d}"
    results_dir = os.path.join(config.RESULTS_DIR, f"TCN_{run_name}_{date_str}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {results_dir}")

    seqs, labels = load_all_jsons(args.directory)
    if len(seqs) == 0:
        print("[ERROR] No valid sequences loaded.")
        return

    X_train, X_test, y_train, y_test = prepare_data(seqs, labels)

    model = create_tcn_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early = EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                   min_lr=1e-6, verbose=1)

    unique_train_classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=unique_train_classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(unique_train_classes, cw)}
    print(f"[INFO] Class weights: {class_weight_dict}")

    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )

    save_training_history(history, results_dir)
    save_learning_curves(history, results_dir)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)
    print(f"[INFO] Model saved to {args.model_path}")

    results_model_path = os.path.join(results_dir, "tcn_model.h5")
    model.save(results_model_path)

    evaluate_model(model, X_test, y_test, results_dir)


# ============================================================
# PREDICTION
# ============================================================

def predict_clip(args):
    if not os.path.isfile(args.file):
        print(f"[ERROR] JSON file not found: {args.file}")
        return -1
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        return -1

    model = load_model(args.model_path)

    from keras.utils import pad_sequences
    import json

    with open(args.file, "r") as f:
        data = json.load(f)

    pelvis   = np.array(data.get(config.FIELD_PELVIS, []))
    shoulder = np.array(data.get(config.FIELD_SHOULDER_RELATIVE, []))
    elbow    = np.array(data.get(config.FIELD_ELBOW_RELATIVE, []))
    hand     = np.array(data.get(config.FIELD_WRIST_RELATIVE, []))

    has_shoulder = len(shoulder) > 0
    if has_shoulder:
        L = min(len(pelvis), len(shoulder), len(elbow), len(hand))
        sequence = np.concatenate([pelvis[:L], shoulder[:L], elbow[:L], hand[:L]], axis=1)
    else:
        L = min(len(pelvis), len(elbow), len(hand))
        sequence = np.concatenate([pelvis[:L], elbow[:L], hand[:L]], axis=1)

    sequence = normalize_sequence(sequence)
    sequence = pad_sequences([sequence], maxlen=config.MAX_SEQUENCE_LENGTH,
                             dtype='float32', padding='post', truncating='post')

    pred_class = np.argmax(model.predict(sequence, verbose=0), axis=1)[0]
    class_label = config.CLASS_LABELS[pred_class] if pred_class < len(config.CLASS_LABELS) else str(pred_class)

    print(f"\n[RESULT] Predicted class: {pred_class} ({class_label})\n")
    return pred_class

"""
-------------------------------------------------------------
Module: module_GRU.py
Author: Cesar Emilio Castaño Marin
Project: Thesis - Smash Vision / GRU for Paddle Tennis Analysis
-------------------------------------------------------------
Bidirectional GRU model used as a comparative baseline against the LSTM.
The GRU simplifies the LSTM by merging the forget and input gates into a
single update gate and eliminating the separate cell state, resulting in
fewer parameters and faster training.
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
    GRU, Dense, Dropout,
    BatchNormalization, Bidirectional
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
from Modules.module_data import (
    load_all_jsons,
    prepare_data,
    evaluate_model,
    normalize_sequence,
    save_training_history,
    save_learning_curves,
    get_next_test_number
)


# ============================================================
# GRU MODEL
# ============================================================

def create_gru_model(input_shape, num_classes=config.NUM_CLASSES):
    reg = l2(config.L2_REGULARIZATION)

    model = Sequential([
        BatchNormalization(),

        Bidirectional(GRU(config.GRU_UNITS_L1, return_sequences=True,
                          kernel_regularizer=reg, recurrent_regularizer=reg)),
        Dropout(config.GRU_DROPOUT_L1),

        Bidirectional(GRU(config.GRU_UNITS_L2,
                          kernel_regularizer=reg, recurrent_regularizer=reg)),
        Dropout(config.GRU_DROPOUT_L2),

        Dense(config.GRU_DENSE_UNITS, activation='relu', kernel_regularizer=reg),
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
# TRAINING
# ============================================================

def train_model(args):
    if not os.path.isdir(args.directory):
        print(f"[ERROR] Directory not found: {args.directory}")
        return

    date_str = datetime.now().strftime("%d-%m-%y")
    date_str_full = datetime.now().strftime("%d-%m-%Y")
    test_num = get_next_test_number("GRU")
    run_name = f"Test{test_num:02d}"
    results_dir = os.path.join(config.RESULTS_DIR, f"GRU_{run_name}_{date_str}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {results_dir}")

    seqs, labels = load_all_jsons(args.directory)
    if len(seqs) == 0:
        print("[ERROR] No valid sequences loaded.")
        return

    X_train, X_test, y_train, y_test = prepare_data(seqs, labels)

    model = create_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))

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

    model_filename = f"gruModel_{run_name}_{date_str_full}.h5"
    model_path = os.path.join(config.MODELS_DIR, model_filename)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    results_model_path = os.path.join(results_dir, model_filename)
    model.save(results_model_path)

    evaluate_model(model, X_test, y_test, results_dir)


# ============================================================
# PREDICTION
# ============================================================

def predict_clip(args):
    if not os.path.isfile(args.file):
        print(f"[ERROR] JSON file not found: {args.file}")
        return -1

    model_path = args.model_path if args.model_path else config.get_latest_model("gruModel")
    if not model_path or not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found. Train a model first or specify --model_path.")
        return -1

    model = load_model(model_path)
    print(f"[INFO] Model loaded from {model_path}")

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

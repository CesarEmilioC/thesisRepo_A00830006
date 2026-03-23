"""
config.py
=========
Central configuration for the Paddle Tennis Movement Feedback System.

All magic numbers, file paths, hyperparameters, and constants are defined here
to avoid scattered hardcoded values across the codebase.

Author: Cesar Emilio Castano Marin
"""

import os

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Absolute path to the project root (one level above Source/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Directory where JSON coordinate files are saved after pose estimation
COORDINATES_DIR = os.path.join(ROOT_DIR, 'Coordinates')

# Directory where training experiment results are saved
RESULTS_DIR = os.path.join(ROOT_DIR, 'Source', 'Results')

# Directory where trained models are stored
MODELS_DIR = os.path.join(ROOT_DIR, 'Source', 'Models')

# Default path for the LSTM model file
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')

# ============================================================
# OPENPOSE KEYPOINT INDICES (Body-25 Model)
# ============================================================
# Reference: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
#
# Body-25 keypoint mapping (subset used in this project):
#   0  = Nose              9  = Left Hip         18 = Left Big Toe
#   1  = Neck              10 = Left Knee        19 = Left Small Toe
#   2  = Right Shoulder    11 = Left Ankle       20 = Left Heel
#   3  = Right Elbow       12 = Right Hip        21 = Right Big Toe
#   4  = Right Wrist       13 = Right Knee       22 = Right Small Toe
#   5  = Left Shoulder *   14 = Right Ankle      23 = Right Heel
#   6  = Left Elbow        15 = Right Eye        24 = Background
#   7  = Left Wrist        16 = Left Eye
#   8  = MidHip (Pelvis) * 17 = Right Ear
#
# * = used in this project
KEYPOINT_PELVIS = 8
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_ELBOW = 3
KEYPOINT_RIGHT_WRIST = 4

# ============================================================
# JSON COORDINATE FIELD NAMES
# ============================================================
# These field names are used as keys in the JSON coordinate files.
FIELD_PELVIS = "Pelvis"
FIELD_SHOULDER_ORIGINAL = "Left Shoulder Original"
FIELD_SHOULDER_RELATIVE = "Left Shoulder Relative"
FIELD_ELBOW_ORIGINAL = "Right Elbow Original"
FIELD_ELBOW_RELATIVE = "Right Elbow Relative"
FIELD_WRIST_ORIGINAL = "Right Wrist Original"
FIELD_WRIST_RELATIVE = "Right Wrist Relative"

# ============================================================
# POSE ESTIMATION DEFAULTS
# ============================================================
DEFAULT_OPENPOSE_MODEL = "mobilenet_thin"
DEFAULT_RESIZE = "0x0"
DEFAULT_RESIZE_OUT_RATIO = 4.0

# ============================================================
# SHARED HYPERPARAMETERS
# ============================================================
NUM_CLASSES = 5               # Number of output classes (5-class grouping)
NUM_FEATURES = 8              # Features per frame: pelvis(2) + shoulder(2) + elbow(2) + wrist(2)
MAX_SEQUENCE_LENGTH = 90      # Maximum frames per clip (pad/truncate to this)
EPOCHS = 100                  # Maximum training epochs
BATCH_SIZE = 16               # Training batch size
EARLY_STOPPING_PATIENCE = 15  # Epochs to wait before early stopping
NORMALIZATION_EPSILON = 1e-8  # Small value to avoid division by zero in normalization
L2_REGULARIZATION = 1e-4      # L2 weight penalty to reduce overfitting
AUGMENTATION_NOISE_STD = 0.02 # Gaussian noise std for data augmentation
AUGMENTATION_FACTOR = 2       # Number of augmented copies per original sample

# ============================================================
# LSTM HYPERPARAMETERS
# ============================================================
# Bidirectional LSTM architecture (reduced to fight overfitting):
#   Input -> BatchNorm -> BiLSTM(64) -> Dropout(0.40)
#         -> BiLSTM(32) -> Dropout(0.35) -> Dense(32) -> BatchNorm
#         -> Dense(NUM_CLASSES, softmax)
LSTM_UNITS_L1 = 64            # Units in first Bidirectional LSTM layer
LSTM_UNITS_L2 = 32            # Units in second Bidirectional LSTM layer
DROPOUT_L1 = 0.40             # Dropout rate after first LSTM layer
DROPOUT_L2 = 0.35             # Dropout rate after second LSTM layer
DENSE_UNITS = 32              # Units in the hidden Dense layer

# ============================================================
# GRU HYPERPARAMETERS
# ============================================================
# Bidirectional GRU architecture:
#   Input -> BatchNorm -> BiGRU(64) -> Dropout(0.40)
#         -> BiGRU(32) -> Dropout(0.35) -> Dense(32) -> BatchNorm
#         -> Dense(NUM_CLASSES, softmax)
GRU_UNITS_L1 = 64
GRU_UNITS_L2 = 32
GRU_DROPOUT_L1 = 0.40
GRU_DROPOUT_L2 = 0.35
GRU_DENSE_UNITS = 32

# ============================================================
# TCN HYPERPARAMETERS
# ============================================================
# Temporal Convolutional Network:
#   Input -> BatchNorm -> Conv1D(64,3,dilation=1) -> Conv1D(64,3,dilation=2)
#         -> Conv1D(64,3,dilation=4) -> GlobalAvgPool -> Dropout(0.40)
#         -> Dense(32) -> Dense(NUM_CLASSES, softmax)
TCN_FILTERS = 64              # Number of filters per convolutional layer
TCN_KERNEL_SIZE = 3           # Kernel size for 1D convolutions
TCN_DILATIONS = [1, 2, 4]     # Dilation rates for the convolutional layers
TCN_DROPOUT = 0.40
TCN_DENSE_UNITS = 32

# ============================================================
# GRADE-TO-CLASS MAPPING (5-class grouping)
# ============================================================
# Original grades are 1-10. The LSTM classifies into 5 groups:
#   Grade 1-2  -> Class 0 (Very Low)
#   Grade 3-4  -> Class 1 (Low)
#   Grade 5-6  -> Class 2 (Medium)
#   Grade 7-8  -> Class 3 (High)
#   Grade 9-10 -> Class 4 (Excellent)
CLASS_LABELS = ["Very Low", "Low", "Medium", "High", "Excellent"]

def grade_to_class(grade: int) -> int:
    """Convert a grade (1-10) to a class index (0-4)."""
    return min((grade - 1) // 2, 4)

# ============================================================
# VISUALIZATION CONSTANTS
# ============================================================
ANIMATION_PLAYBACK_SPEED = 0.2   # Playback speed multiplier (0.2 = 20% of real speed)
COORDINATE_MARGIN_RATIO = 0.2    # Margin around coordinate bounds in plots (20%)
COLORMAP_RANGE_START = 0.3       # Start of colormap range (lighter = movement start)

# ============================================================
# FILE NAMING CONVENTION
# ============================================================
# Expected filename format: player{N}_part{M}_clip{K}_grade{G}
# Example: player10_part1_clip3_grade8
FILENAME_REGEX = r'player(\d+)_part(\d+)_clip(\d+)_grade(\d+)'

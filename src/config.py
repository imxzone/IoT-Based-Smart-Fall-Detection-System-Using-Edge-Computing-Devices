from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_DIR = PROJECT_ROOT/ "dataset"

RAW_SISFALL_DIR = DATASET_DIR / "sisfall_raw"
PROCESSED_SISFALL_DIR = DATASET_DIR / "sisfall_processed"
RAW_SELF_COLLECT_DIR = DATASET_DIR / "self_collect_raw"
PROCESSED_SELF_COLLECT_DIR = DATASET_DIR / "self_collect_processed"

OUTPUT_DIR = DATASET_DIR / "processed_data"
MODEL_DIR = DATASET_DIR / "models"

# Sampling config
SISFALL_FS = 200
TARGET_FS = 25

# SisFall unit covnersion
ACC_SCALE = (2*16) / (2**13)    # ADXL345 | ±16g 13 bits
GYRO_SCALE = 4000 / (2**16)     # ITG3200 | ±2000 16 bits

# Trial duration (seconds)
TRIAL_DURATION = {
    **{f"D{i:02d}": 100 for i in range(1, 5)},
    **{f"D{i:02d}": 25 for i in range(5, 7)},
    **{f"D{i:02d}": 16 for i in range(7, 20)},
    **{f"F{i:02d}": 16 for i in range(1, 16)},
}

# Window config
WINDOW_SECONDS = 10
WINDOW_STEP_SECONDS = 2

WINDOW_SIZE = WINDOW_SECONDS * TARGET_FS
STEP_SIZE = WINDOW_STEP_SECONDS * TARGET_FS

# Real-time trigger
AVM_THRESHOLD_G = 2.5

# Feture extraction
FFT_LENGTH = 32

INERTIAL_SIGNAL_COLS = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "avm",
]

PRESSURE_SIGNAL_COLS = [
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "avm", "pressure",
]

LOW_FREQ_RATIO = 10

# Dataset config
DATASET_CONFIGS = {
    "sisfall": {"signal_cols": INERTIAL_SIGNAL_COLS,},

    "self_no_pressure": {"signal_cols": INERTIAL_SIGNAL_COLS,},

    "self_with_pressure": {"signal_cols": PRESSURE_SIGNAL_COLS,},

    "combined": {"signal_cols": INERTIAL_SIGNAL_COLS,},
}

# Other
RANDOM_STATE = 43


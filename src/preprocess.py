import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from src.config import ACC_SCALE, GYRO_SCALE, TARGET_FS, SISFALL_FS, INERTIAL_SIGNAL_COLS, PRESSURE_SIGNAL_COLS
from src.io_utils import read_sisfall


def convert_sisfall_units(df: pd.DataFrame):
    out = pd.DataFrame()

    out["ax"] = df["acc1_x_raw"] * ACC_SCALE
    out["ay"] = df["acc1_y_raw"] * ACC_SCALE
    out["az"] = df["acc1_z_raw"] * ACC_SCALE

    out["gx"] = df["gyro_x_raw"] * GYRO_SCALE
    out["gy"] = df["gyro_y_raw"] * GYRO_SCALE
    out["gz"] = df["gyro_z_raw"] * GYRO_SCALE

    return out

def compute_avm(df: pd.DataFrame):
    df = df.copy()
    df["avm"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)

    return df


def downsample_to_25hz(df: pd.DataFrame):
    signal_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    downsampled = {}

    for col in signal_cols:
        downsampled[col] = resample_poly(
            df[col].to_numpy(),
            up = TARGET_FS,
            down = SISFALL_FS,
        )
    df_25hz = pd.DataFrame(downsampled)

    return df_25hz


def process_sisfall_file(file_path):
    raw_df = read_sisfall(file_path)

    df = convert_sisfall_units(raw_df)
    df = downsample_to_25hz(df)
    df = compute_avm(df)

    return df[INERTIAL_SIGNAL_COLS]

import re
from pathlib import Path
import pandas as pd


def parse_filename(file_path: Path):
    match = re.fullmatch(r"([DF]\d{2})_((?:SA|SE|NG|OG)\d{2})_(R\d{2})\.txt", file_path.name)

    if match is None:
        raise ValueError(f"Invalid filename format: {file_path.name}")

    activity, subject, trial = match.groups()
    label = 1 if activity.startswith("F") else 0

    return subject, activity, trial, label

def validate_filename(file_path: Path):
    valid = re.fullmatch(r"[DF]\d{2}_((?:SA|SE|NG|OG)\d{2})_R\d{2}\.txt", file_path.name) is not None

    if not valid:
        print(f"Invalid filename: {file_path.name}")

    return valid

def normalize_txt_extensions(dataset_dir: Path):
    for file_path in dataset_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() == ".txt" and file_path.suffix != ".txt":
            file_path.rename(file_path.with_suffix(".txt"))

def read_sisfall(file_path: Path):
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            line = line.replace(";", "")
            parts = [part.strip() for part in line.split(",") if part.strip()]

            if len(parts) != 9:
                continue

            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid SisFall samples found in: {file_path.name}")

    return pd.DataFrame(rows, columns=[
        "acc1_x_raw", "acc1_y_raw", "acc1_z_raw", 
        "gyro_x_raw", "gyro_y_raw", "gyro_z_raw", 
        "acc2_x_raw", "acc2_y_raw", "acc2_z_raw"])
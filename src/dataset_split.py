import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.config import RANDOM_STATE, PROCESSED_SISFALL_DIR, PROCESSED_SELF_COLLECT_DIR, OUTPUT_DIR
from src.io_utils import parse_filename


def collect_metadata(dataset_name):
    sources = {
        "sisfall": [("sisfall", PROCESSED_SISFALL_DIR)],
        "self_collect": [("self_collect", PROCESSED_SELF_COLLECT_DIR)],
        "combine": [("sisfall", PROCESSED_SISFALL_DIR), ("self_collect", PROCESSED_SELF_COLLECT_DIR)],
    }

    if dataset_name not in sources:
        raise ValueError("dataset_name must be 'sisfall', 'self_collect', or 'combine'")

    metadata = []

    for source, folder in sources[dataset_name]:
        for file_path in sorted(folder.rglob("*.txt")):
            subject, activity, trial, label = parse_filename(file_path)

            metadata.append({
                "file_path": str(file_path),
                "source": source,
                "subject": subject,
                "activity": activity,
                "trial": trial,
                "label": label,
            })

    if not metadata:
        raise ValueError(f"No processed files found for dataset: {dataset_name}")

    metadata_df = pd.DataFrame(metadata).sort_values(["source", "subject", "activity", "trial"]).reset_index(drop=True)

    duplicate_cols = ["source", "subject", "activity", "trial"]
    duplicates = metadata_df[metadata_df.duplicated(subset=duplicate_cols, keep=False)]

    if not duplicates.empty:
        raise ValueError("Duplicate recordings found:\n" + duplicates[duplicate_cols + ["file_path"]].to_string(index=False))

    return metadata_df


def split_dataset(dataset_name):
    metadata_df = collect_metadata(dataset_name)
    metadata_df["split"] = ""

    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_valid_idx, test_idx = next(outer_splitter.split(X=metadata_df, y=metadata_df["label"], groups=metadata_df["subject"]))

    metadata_df.loc[test_idx, "split"] = "test"
    train_valid_df = metadata_df.iloc[train_valid_idx].copy()

    inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    train_idx, valid_idx = next(inner_splitter.split(X=train_valid_df, y=train_valid_df["label"], groups=train_valid_df["subject"]))

    metadata_df.loc[train_valid_df.iloc[train_idx].index, "split"] = "train"
    metadata_df.loc[train_valid_df.iloc[valid_idx].index, "split"] = "valid"

    train_df = metadata_df[metadata_df["split"] == "train"]
    valid_df = metadata_df[metadata_df["split"] == "valid"]
    test_df = metadata_df[metadata_df["split"] == "test"]

    total_files = len(metadata_df)
    train_subjects = sorted(train_df["subject"].unique().tolist())
    valid_subjects = sorted(valid_df["subject"].unique().tolist())
    test_subjects = sorted(test_df["subject"].unique().tolist())

    train_ratio = len(train_df) / total_files * 100
    valid_ratio = len(valid_df) / total_files * 100
    test_ratio = len(test_df) / total_files * 100

    print()
    print(f"Dataset: {dataset_name}")
    print(f"Total files: {total_files}")
    print(f"Total subjects: {metadata_df['subject'].nunique()}")
    print("Target: Train 64% | Valid 16% | Test 20%")
    print(f"Train: {len(train_df)} files ({train_ratio:.2f}%) | {len(train_subjects)} subjects | {train_subjects}")
    print(f"Valid: {len(valid_df)} files ({valid_ratio:.2f}%) | {len(valid_subjects)} subjects | {valid_subjects}")
    print(f"Test:  {len(test_df)} files ({test_ratio:.2f}%) | {len(test_subjects)} subjects | {test_subjects}")

    split_info = {
        "dataset": dataset_name,
        "target_ratio": {"train": 64, "valid": 16, "test": 20},
        "actual_ratio": {"train": round(train_ratio, 2), "valid": round(valid_ratio, 2), "test": round(test_ratio, 2)},
        "total_files": total_files,
        "total_subjects": metadata_df["subject"].nunique(),
        "train_subjects": train_subjects,
        "valid_subjects": valid_subjects,
        "test_subjects": test_subjects,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"split_{dataset_name}.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=4)

    print(f"Saved JSON: {json_path}")

    return metadata_df
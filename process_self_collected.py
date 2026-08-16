import pandas as pd

from src.config import (
    RAW_SELF_COLLECT_DIR,
    PROCESSED_SELF_COLLECT_DIR,
    PRESSURE_SIGNAL_COLS,
    TRIAL_DURATION,
    TARGET_FS,
)


def read_self_collected_file(file_path):
    df = pd.read_csv(file_path)

    # Already contains only the 8 required signals
    if list(df.columns) == PRESSURE_SIGNAL_COLS:
        return df[PRESSURE_SIGNAL_COLS]

    # Original STM32 raw format
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        next(file, None)

        for line in file:
            parts = line.strip().split(",")

            if len(parts) < 10:
                continue

            try:
                time_ms = int(parts[1])

                if time_ms % 40 != 0:
                    continue

                signals = [float(value) for value in parts[2:10]]
                rows.append(signals)

            except ValueError:
                continue

    return pd.DataFrame(rows, columns=PRESSURE_SIGNAL_COLS)


def get_activity_code(file_path):
    return file_path.stem.split("_")[0]


def get_expected_samples(activity_code):
    duration = TRIAL_DURATION[activity_code]

    return duration * TARGET_FS


def main():
    total_processed = 0
    total_skipped = 0

    for subject_id in range(1, 51):
        subject = f"NG{subject_id:02d}"

        input_dir = RAW_SELF_COLLECT_DIR / subject
        output_dir = PROCESSED_SELF_COLLECT_DIR / subject

        if not input_dir.exists():
            print(f"[SKIP] {subject}: folder not found")
            continue

        files = [
            file
            for file in input_dir.rglob("*")
            if file.is_file() and file.suffix.lower() == ".txt"
        ]

        print()
        print(f"{subject}: found {len(files)} files")

        for input_path in files:
            try:
                activity_code = get_activity_code(input_path)

                if activity_code not in TRIAL_DURATION:
                    print(f"[SKIP] {input_path.name}: unknown activity")
                    total_skipped += 1
                    continue

                df = read_self_collected_file(input_path)

                expected_samples = get_expected_samples(activity_code)

                if len(df) < expected_samples:
                    print(
                        f"[SKIP] {input_path.name}: "
                        f"{len(df)}/{expected_samples} samples"
                    )
                    total_skipped += 1
                    continue

                df = df.iloc[:expected_samples].reset_index(drop=True)

                relative_path = input_path.relative_to(input_dir)
                output_path = output_dir / relative_path

                output_path.parent.mkdir(parents=True, exist_ok=True)

                df.to_csv(
                    output_path,
                    index=False,
                )

                total_processed += 1

                print(
                    f"[OK] {input_path.name}: "
                    f"{len(df)} samples"
                )

            except Exception as e:
                print(f"[ERROR] {input_path.name}: {e}")
                total_skipped += 1

    print()
    print("=" * 50)
    print(f"Total processed: {total_processed}")
    print(f"Total skipped:   {total_skipped}")


if __name__ == "__main__":
    main()
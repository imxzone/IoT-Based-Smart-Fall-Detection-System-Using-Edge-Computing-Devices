from src.config import RAW_SISFALL_DIR, PROCESSED_SISFALL_DIR
from src.preprocess import process_sisfall_file


def main():
    files = []

    for file in RAW_SISFALL_DIR.rglob("*"):
        if file.is_file() and file.suffix.lower() == ".txt":
            files.append(file)

    print(f"Found {len(files)} SisFall files")

    for input_path in files:
        try:
            df = process_sisfall_file(input_path)
            relative_path = input_path.relative_to(RAW_SISFALL_DIR)
            output_path = PROCESSED_SISFALL_DIR / relative_path

            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_path, index=False)

            print(f"[OK] {input_path.name}: {len(df)} samples")

        except Exception as e:
            print(f"[ERROR] {input_path.name}: {e}")


if __name__ == "__main__":
    main()
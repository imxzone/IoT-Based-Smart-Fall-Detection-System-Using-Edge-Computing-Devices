from src.config import DATASET_DIR
from src.io_utils import normalize_txt_extensions, validate_filename


def check_self_dataset():
    normalize_txt_extensions(DATASET_DIR)

    total_files = 0
    invalid_files = 0

    for file_path in DATASET_DIR.rglob("*.txt"):
        total_files += 1

        if not validate_filename(file_path):
            invalid_files += 1

    print()
    print(f"Total files: {total_files}")
    print(f"Valid files: {total_files - invalid_files}")
    print(f"Invalid files: {invalid_files}")


if __name__ == "__main__":
    check_self_dataset()
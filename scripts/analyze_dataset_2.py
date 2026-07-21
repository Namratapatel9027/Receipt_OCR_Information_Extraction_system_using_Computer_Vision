from pathlib import Path
import json
import cv2


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw"

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"


# ==========================================================
# Helper Functions
# ==========================================================

def count_files(folder: Path) -> int:
    """Return total number of files in a folder."""
    return len([f for f in folder.iterdir() if f.is_file()])


def get_image_info(image_path: Path):
    """Return width, height and channels of an image."""
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width, channels = image.shape
    return width, height, channels


def read_ocr_annotation(file_path: Path):
    """Read OCR annotation file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def read_entity_annotation(file_path: Path):
    """Read entity annotation JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n" + "=" * 60)
    print("INTELLIGENT RECEIPT PROCESSING SYSTEM (IRPS)")
    print("DATASET ANALYZER")
    print("=" * 60)

    # ------------------------------------------------------
    # Dataset Summary
    # ------------------------------------------------------

    train_img = TRAIN_DIR / "img"
    train_box = TRAIN_DIR / "box"
    train_entities = TRAIN_DIR / "entities"

    test_img = TEST_DIR / "img"
    test_box = TEST_DIR / "box"
    test_entities = TEST_DIR / "entities"

    print_section("DATASET SUMMARY")

    print(f"Train Images       : {count_files(train_img)}")
    print(f"Train OCR Files    : {count_files(train_box)}")
    print(f"Train Entity Files : {count_files(train_entities)}\n")

    print(f"Test Images        : {count_files(test_img)}")
    print(f"Test OCR Files     : {count_files(test_box)}")
    print(f"Test Entity Files  : {count_files(test_entities)}")

    # ------------------------------------------------------
    # Sample Image
    # ------------------------------------------------------

    sample_image = sorted(train_img.glob("*.jpg"))[0]

    image_info = get_image_info(sample_image)

    print_section("SAMPLE IMAGE")

    print(f"Filename : {sample_image.name}")

    if image_info:
        width, height, channels = image_info
        print(f"Width    : {width}")
        print(f"Height   : {height}")
        print(f"Channels : {channels}")

    # ------------------------------------------------------
    # Sample OCR Annotation
    # ------------------------------------------------------

    sample_box = train_box / f"{sample_image.stem}.txt"

    ocr_lines = read_ocr_annotation(sample_box)

    print_section("SAMPLE OCR ANNOTATION")

    print(f"Filename        : {sample_box.name}")
    print(f"Total OCR Lines : {len(ocr_lines)}\n")

    print("First 5 OCR Entries:\n")

    for i, line in enumerate(ocr_lines[:5], start=1):
        print(f"{i}. {line}")

    # ------------------------------------------------------
    # Sample Entity Annotation
    # ------------------------------------------------------

    sample_entity = train_entities / f"{sample_image.stem}.txt"

    entity = read_entity_annotation(sample_entity)

    print_section("SAMPLE ENTITY ANNOTATION")

    print(f"Filename : {sample_entity.name}\n")

    for key, value in entity.items():
        print(f"{key:<10}: {value}")

    # ------------------------------------------------------
    # Completed
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
from pathlib import Path
from PIL import Image


def get_stems(folder: Path, extension: str):
    return {file.stem for file in folder.glob(f"*{extension}")}


def check_corrupted_images(img_dir: Path):
    corrupted = []

    for image_path in img_dir.glob("*.jpg"):
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupted.append(image_path.name)

    return corrupted


def verify_split(split_path: Path, split_name: str):
    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET VERIFICATION")
    print("=" * 70)

    img_dir = split_path / "img"
    box_dir = split_path / "box"
    entity_dir = split_path / "entities"

    # ---------------- Folder Check ---------------- #

    for folder in [img_dir, box_dir, entity_dir]:
        if not folder.exists():
            print(f"❌ Missing Folder : {folder}")
            return

    print("✅ Required folders found.")

    # ---------------- Count Check ---------------- #

    image_files = list(img_dir.glob("*.jpg"))
    box_files = list(box_dir.glob("*.txt"))
    entity_files = list(entity_dir.glob("*.txt"))

    print("\nFile Counts")
    print("-" * 30)
    print(f"Images   : {len(image_files)}")
    print(f"Boxes    : {len(box_files)}")
    print(f"Entities : {len(entity_files)}")

    if len(image_files) == len(box_files) == len(entity_files):
        print("✅ File count check passed.")
    else:
        print("❌ File count mismatch.")

    # ---------------- Filename Check ---------------- #

    image_names = get_stems(img_dir, ".jpg")
    box_names = get_stems(box_dir, ".txt")
    entity_names = get_stems(entity_dir, ".txt")

    missing_boxes = sorted(image_names - box_names)
    missing_entities = sorted(image_names - entity_names)

    extra_boxes = sorted(box_names - image_names)
    extra_entities = sorted(entity_names - image_names)

    print("\nFilename Verification")
    print("-" * 30)

    if (
        not missing_boxes
        and not missing_entities
        and not extra_boxes
        and not extra_entities
    ):
        print("✅ All filenames match.")
    else:
        if missing_boxes:
            print(f"❌ Missing Box Files : {len(missing_boxes)}")

        if missing_entities:
            print(f"❌ Missing Entity Files : {len(missing_entities)}")

        if extra_boxes:
            print(f"❌ Extra Box Files : {len(extra_boxes)}")

        if extra_entities:
            print(f"❌ Extra Entity Files : {len(extra_entities)}")

    # ---------------- Corrupted Images ---------------- #

    print("\nImage Verification")
    print("-" * 30)

    corrupted = check_corrupted_images(img_dir)

    if corrupted:
        print(f"❌ Corrupted Images : {len(corrupted)}")

        for img in corrupted[:10]:
            print(img)

    else:
        print("✅ No corrupted images found.")

    # ---------------- Summary ---------------- #

    print("\nSummary")
    print("-" * 30)

    if (
        len(image_files) == len(box_files) == len(entity_files)
        and not missing_boxes
        and not missing_entities
        and not extra_boxes
        and not extra_entities
        and not corrupted
    ):
        print("🎉 Dataset verification PASSED.")
    else:
        print("⚠ Dataset verification FAILED.")


def main():
    project_root = Path(__file__).resolve().parent.parent

    raw_dir = project_root / "data" / "raw"

    verify_split(raw_dir / "train", "Train")
    verify_split(raw_dir / "test", "Test")


if __name__ == "__main__":
    main()
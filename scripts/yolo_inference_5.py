"""Self-contained YOLO inference script for receipt field localization, cropping, and visualization.

Consolidates YOLO detector, cropping, and visualization modules into a single CLI execution script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Output Directory Configuration
OUTPUT_ROOT = Path("outputs")
CROPS_DIR = OUTPUT_ROOT / "yolo_crops"
VISUALIZATION_DIR = OUTPUT_ROOT / "yolo_visualization"

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_MAPPING = {"company": 0, "date": 1, "address": 2, "total": 3}
CLASS_COLORS = {
    "company": (0, 0, 255),      # Red
    "date": (0, 255, 0),         # Green
    "address": (255, 0, 0),       # Blue
    "total": (0, 255, 255),       # Yellow
}


def get_logger(name: str) -> logging.Logger:
    """Initialize a standardized, structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Detection hyperparameter and hardware configuration."""

    model_path: Path
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class Detection:
    """Standardized bounding box detection output from YOLO."""

    class_id: int
    class_name: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize values for output generation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box_xyxy": [self.xmin, self.ymin, self.xmax, self.ymax],
        }


class ReceiptDetector:
    """Loads weights and runs field detection on input receipt images."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        if not config.model_path.is_file():
            raise FileNotFoundError(f"YOLO weights file not found: {config.model_path}")

        try:
            import cv2
            from ultralytics import YOLO

            self._cv2 = cv2
            self.model = YOLO(str(config.model_path))
        except ImportError as error:
            raise ImportError(
                "Missing packages (ultralytics or opencv-python). Please run: "
                "pip install ultralytics opencv-python"
            ) from error
        except Exception as error:
            raise RuntimeError(f"Failed to load YOLO model: {error}") from error

    def load_image(self, path: str | Path) -> tuple[Path, Any]:
        """Read and validate the source image file."""
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Target receipt image not found: {resolved_path}")
        image = self._cv2.imread(str(resolved_path), self._cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"OpenCV could not read target image: {resolved_path}")
        return resolved_path, image

    def predict(self, image_path: str | Path) -> tuple[Path, Any, list[Detection]]:
        """Run YOLO inference and filter duplicate classes and overlaps."""
        path, image = self.load_image(image_path)
        try:
            results = self.model.predict(
                source=image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.image_size,
                device=self.config.device,
                verbose=self.config.verbose,
            )
        except Exception as error:
            raise RuntimeError(f"YOLO inference failed for: {path}") from error

        if not results:
            self.logger.warning("YOLO returned no result object for %s", path.name)
            return path, image, []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self.logger.info("No fields detected in %s", path.name)
            return path, image, []

        image_height, image_width = image.shape[:2]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = str(names.get(class_id, class_id))
            xmin, ymin, xmax, ymax = (int(value) for value in box.xyxy[0].tolist())
            xmin = max(0, min(xmin, image_width))
            ymin = max(0, min(ymin, image_height))
            xmax = max(xmin, min(xmax, image_width))
            ymax = max(ymin, min(ymax, image_height))
            if xmax == xmin or ymax == ymin:
                continue
            detections.append(
                Detection(class_id, class_name, confidence, xmin, ymin, xmax, ymax)
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)

        accepted: list[Detection] = []
        seen_classes: set[str] = set()

        for candidate in detections:
            # 1. Class Deduplication
            if candidate.class_name in seen_classes:
                self.logger.info(
                    "Filtered duplicate class '%s' with lower confidence %f in %s",
                    candidate.class_name,
                    candidate.confidence,
                    path.name,
                )
                continue

            # 2. Overlap suppression (> 20% Intersection over Minimum Area)
            overlaps = False
            candidate_area = (candidate.xmax - candidate.xmin) * (
                candidate.ymax - candidate.ymin
            )
            for acc in accepted:
                x_left = max(candidate.xmin, acc.xmin)
                y_top = max(candidate.ymin, acc.ymin)
                x_right = min(candidate.xmax, acc.xmax)
                y_bottom = min(candidate.ymax, acc.ymax)

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    acc_area = (acc.xmax - acc.xmin) * (acc.ymax - acc.ymin)
                    min_area = min(candidate_area, acc_area)
                    overlap_ratio = intersection_area / min_area

                    if overlap_ratio > 0.20:
                        self.logger.info(
                            "Filtered overlapping box '%s' (%f overlap ratio with '%s') in %s",
                            candidate.class_name,
                            overlap_ratio,
                            acc.class_name,
                            path.name,
                        )
                        overlaps = True
                        break

            if not overlaps:
                accepted.append(candidate)
                seen_classes.add(candidate.class_name)

        self.logger.info(
            "Detected %d fields in %s (kept from %d raw boxes)",
            len(accepted),
            path.name,
            len(detections),
        )
        return path, image, accepted


def save_detection_crops(
    image: Any,
    receipt_id: str,
    detections: Sequence[Detection],
    output_root: Path = CROPS_DIR,
) -> list[Path]:
    """Slice crop regions from the source image and save them to disk."""
    import cv2

    receipt_dir = output_root / receipt_id
    receipt_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    # Index prefix ensures alphabetical sorting matches typical processing order
    for index, detection in enumerate(detections, start=1):
        crop = image[
            detection.ymin : detection.ymax, detection.xmin : detection.xmax
        ]
        filename = f"{index:02d}_{detection.class_name}.jpg"
        crop_path = (receipt_dir / filename).resolve()
        if cv2.imwrite(str(crop_path), crop):
            saved_paths.append(crop_path)
        else:
            raise OSError(f"Could not save cropped image to {crop_path}")

    get_logger("yolo_cropper").info(
        "Saved %d field crops to %s", len(saved_paths), receipt_dir
    )
    return saved_paths


def save_visualization(
    image: Any,
    source_stem: str,
    detections: Sequence[Detection],
    output_dir: Path = VISUALIZATION_DIR,
) -> Path:
    """Annotate the original receipt image with colored boxes and save it."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    for detection in detections:
        color = CLASS_COLORS.get(detection.class_name.lower(), (255, 255, 255))
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        cv2.rectangle(
            canvas,
            (detection.xmin, detection.ymin),
            (detection.xmax, detection.ymax),
            color,
            2,
        )
        text_y = max(18, detection.ymin - 7)
        cv2.putText(
            canvas,
            label,
            (detection.xmin, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path = (output_dir / f"{source_stem}_detections.jpg").resolve()
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Unable to write visualization: {output_path}")
    get_logger("yolo_visualize").info(
        "Saved detection visualization to %s", output_path
    )
    return output_path


def run_single_inference(
    detector: ReceiptDetector,
    image_path: Path,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Execute detection, crop extraction, and visualization for one image."""
    image_path, image, detections = detector.predict(image_path)
    receipt_id = image_path.stem
    crop_paths = (
        []
        if arguments.no_crops
        else save_detection_crops(image, receipt_id, detections)
    )
    visualization_path = None
    if not arguments.no_visualization:
        visualization_path = save_visualization(image, receipt_id, detections)

    return {
        "receipt_id": receipt_id,
        "source_image": str(image_path),
        "detection_count": len(detections),
        "crop_paths": [str(path) for path in crop_paths],
        "visualization_path": str(visualization_path) if visualization_path else None,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI orchestrator for the merged YOLO inference pipeline."""
    parser = argparse.ArgumentParser(description="Run receipt field detection.")
    parser.add_argument(
        "image",
        type=Path,
        help="Path to one receipt image or a directory of receipt images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained YOLO best.pt weights.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run YOLO model on (cpu, cuda, etc.).",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Do not save crop images.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Do not save annotated detection visualization.",
    )

    parsed = parser.parse_args(arguments)
    logger = get_logger("YoloPipelineMain")

    yolo_model_path = parsed.model
    if not yolo_model_path:
        script_dir = Path(__file__).resolve().parent
        yolo_model_path = (script_dir.parent / "models" / "best.pt").resolve()

    cfg = InferenceConfig(
        model_path=yolo_model_path,
        confidence_threshold=parsed.confidence,
        iou_threshold=parsed.iou,
        device=parsed.device,
    )

    try:
        detector = ReceiptDetector(cfg)
        input_path = parsed.image.expanduser().resolve()

        if input_path.is_dir():
            image_paths = sorted(
                p
                for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
            )
            if not image_paths:
                logger.warning("No supported images found in: %s", input_path)
                return 0

            logger.info("Processing %d images in batch...", len(image_paths))
            success_count = 0
            for img_path in image_paths:
                try:
                    run_single_inference(detector, img_path, parsed)
                    success_count += 1
                except Exception as error:
                    logger.error("Failed processing %s: %s", img_path.name, error)

            print(
                json.dumps(
                    {
                        "processed_count": success_count,
                        "total_count": len(image_paths),
                        "message": "YOLO inference complete.",
                    },
                    indent=2,
                )
            )
        else:
            result = run_single_inference(detector, input_path, parsed)
            print(json.dumps(result, indent=2))

        return 0
    except Exception as error:
        logger.exception("Inference failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

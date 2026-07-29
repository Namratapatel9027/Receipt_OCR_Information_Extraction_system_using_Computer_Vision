"""Educational and debugging script to visualize OCR preprocessing variants.

Generates side-by-side crop comparisons (original, sharpened, thresholded)
labeled with PaddleOCR's output text and confidence scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# Output Directory Configuration
OUTPUT_ROOT = Path("outputs")
VARIANTS_VIS_DIR = OUTPUT_ROOT / "ocr_variants_visualization"

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_MAPPING = {"company": 0, "date": 1, "address": 2, "total": 3}


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


class receipt_yolo_detector:
    """Loads YOLO model and runs field detection."""

    def __init__(self, model_path: Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.logger = get_logger("YoloDetector")

    def detect(self, image_path: Path) -> tuple[Any, list[dict[str, Any]]]:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image {image_path}")

        results = self.model.predict(image, conf=0.25, iou=0.45, verbose=False)
        if not results:
            return image, []

        result = results[0]
        if result.boxes is None:
            return image, []

        names = result.names
        detections = []
        image_h, image_w = image.shape[:2]

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = str(names.get(class_id, class_id))
            xmin, ymin, xmax, ymax = (int(val) for val in box.xyxy[0].tolist())

            xmin = max(0, min(xmin, image_w))
            ymin = max(0, min(ymin, image_h))
            xmax = max(xmin, min(xmax, image_w))
            ymax = max(ymin, min(ymax, image_h))

            if xmax > xmin and ymax > ymin:
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": (xmin, ymin, xmax, ymax),
                    }
                )

        # De-duplicate: Keep highest confidence per class
        seen = set()
        filtered = []
        # Sort by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        for det in detections:
            if det["class_name"] not in seen:
                filtered.append(det)
                seen.add(det["class_name"])

        return image, filtered


class ocr_variant_visualizer:
    """Runs PaddleOCR on crops and generates comparative grid visualizations."""

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR

        self.engine = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        self.logger = get_logger("OcrProcessor")

    def quality_metrics(self, image: Any) -> float:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_variants(self, image: Any) -> list[tuple[str, Any]]:
        import cv2

        variants = [("original", image)]

        # 1. Contrast-Sharpened (CLAHE + Gaussian Weighted addition)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (0, 0), 2)
        sharpened = cv2.addWeighted(clahe, 1.5, blurred, -0.5, 0)
        variants.append(("contrast_sharpened", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))

        # 2. Adaptive Thresholded
        thresholded = cv2.adaptiveThreshold(
            clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
        )
        variants.append(("adaptive_threshold", cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)))

        return variants

    def run_ocr(self, image: Any) -> tuple[str, float]:
        """Run PaddleOCR and collect tokens and confidence."""
        result = list(self.engine.predict(image))
        texts: list[str] = []
        scores: list[float] = []

        def collect(value):
            if isinstance(value, str):
                try:
                    collect(json.loads(value))
                except json.JSONDecodeError:
                    return
            elif isinstance(value, dict):
                text_val = value.get("rec_text") or value.get("text")
                score_val = value.get("rec_score") or value.get("score")
                if isinstance(text_val, str) and isinstance(score_val, (int, float)):
                    texts.append(text_val)
                    scores.append(float(score_val))
                for k, v in value.items():
                    if k not in {"rec_texts", "rec_scores"}:
                        collect(v)
            elif isinstance(value, (list, tuple)):
                if (
                    len(value) == 2
                    and isinstance(value[0], str)
                    and isinstance(value[1], (int, float))
                ):
                    texts.append(value[0])
                    scores.append(float(value[1]))
                else:
                    for item in value:
                        collect(item)

        collect(result)
        text = " ".join(t.strip() for t in texts if t.strip()).strip()
        avg_score = sum(scores) / len(scores) if texts and scores else 0.0
        return text, avg_score

    def create_comparison_grid(
        self,
        field_name: str,
        crop_image: Any,
        receipt_id: str,
    ) -> Path:
        import cv2
        import numpy as np

        variants = self.get_variants(crop_image)
        sharpness = self.quality_metrics(crop_image)

        panels = []
        # Max panel width to keep things readable, base panel size on input
        orig_h, orig_w = crop_image.shape[:2]
        
        # Enforce minimum sizes for visualization clarity
        target_h = max(70, orig_h)
        target_w = max(350, orig_w)

        for var_name, var_img in variants:
            # Resize variant to uniform height/width for horizontal concatenation
            resized = cv2.resize(var_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Run OCR on this variant
            ocr_text, ocr_conf = self.run_ocr(var_img)

            # Pad panel at bottom by 75 pixels for text description (light gray border)
            padded = cv2.copyMakeBorder(
                resized,
                0,
                85,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )

            # Annotate variant panel
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Text line 1: Variant Title
            cv2.putText(
                padded,
                f"Variant: {var_name}",
                (10, target_h + 20),
                font,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            # Text line 2: OCR Output
            clean_text = ocr_text if ocr_text else "[No Text Recognized]"
            # Truncate text if too long for panel width
            if len(clean_text) > 35:
                clean_text = clean_text[:32] + "..."
            cv2.putText(
                padded,
                f"OCR: {clean_text}",
                (10, target_h + 40),
                font,
                0.45,
                (255, 0, 0) if ocr_text else (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            # Text line 3: Confidence & Sharpness
            cv2.putText(
                padded,
                f"Conf: {ocr_conf:.2f} | Sharp: {sharpness:.1f}",
                (10, target_h + 60),
                font,
                0.4,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

            panels.append(padded)

        # Concatenate panels side-by-side
        grid = cv2.hconcat(panels)

        # Create output directory
        receipt_dir = VARIANTS_VIS_DIR / receipt_id
        receipt_dir.mkdir(parents=True, exist_ok=True)

        output_path = receipt_dir / f"{field_name}_comparison.jpg"
        cv2.imwrite(str(output_path), grid)
        self.logger.info("Saved visual grid comparison: %s", output_path)
        return output_path


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize OCR crop variants and text recognition outputs."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to receipt image.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained YOLO best.pt weights.",
    )
    parsed = parser.parse_args(arguments)
    logger = get_logger("VisMain")

    yolo_model_path = parsed.model
    if not yolo_model_path:
        script_dir = Path(__file__).resolve().parent
        yolo_model_path = (script_dir.parent / "models" / "best.pt").resolve()

    try:
        detector = receipt_yolo_detector(yolo_model_path)
        visualizer = ocr_variant_visualizer()

        image_path = parsed.image_path.expanduser().resolve()
        if not image_path.is_file():
            logger.error("File not found: %s", image_path)
            return 1

        logger.info("Running YOLO detector on %s...", image_path.name)
        image, detections = detector.detect(image_path)

        if not detections:
            logger.warning("No fields detected on receipt.")
            return 0

        receipt_id = image_path.stem
        logger.info("Generating side-by-side preprocessing comparisons...")

        for det in detections:
            field = det["class_name"]
            box = det["box"]
            # Slices crop from image array
            crop = image[box[1] : box[3], box[0] : box[2]]
            visualizer.create_comparison_grid(field, crop, receipt_id)

        logger.info(
            "All field comparisons saved to folder: %s",
            VARIANTS_VIS_DIR / receipt_id,
        )
        return 0
    except Exception as error:
        logger.exception("Visualization script failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

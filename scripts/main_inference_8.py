"""Master self-contained production pipeline for end-to-end receipt extraction.

This script contains all YOLO, PaddleOCR, and Business Validation components in a
single file. It requires only standard python libraries and third-party dependencies:
- opencv-python
- numpy
- ultralytics
- paddleocr
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import re
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Sequence

# Output Paths (Creates only 2 folders under outputs/ directory)
OUTPUT_ROOT = Path("outputs")
VISUALIZATION_DIR = OUTPUT_ROOT / "yolo_visualization"
JSON_DIR = OUTPUT_ROOT / "ocr_final_json"

# Settings & Classes Configuration
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


# ----------------------------------------------------------------------
# Exceptions & Configuration Dataclasses
# ----------------------------------------------------------------------
class IRPSError(Exception):
    """Base exception class for all IRPS components."""


class ModelLoadError(IRPSError):
    """Raised when model weights or network architecture cannot be initialized."""


class OcrProcessingError(IRPSError):
    """Raised when OCR operations fail."""


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
class OcrConfig:
    """PaddleOCR system settings and gate values."""

    language: str = "en"
    minimum_confidence: float = 0.75
    minimum_sharpness: float = 35.0


@dataclass(frozen=True, slots=True)
class BusinessRulesConfig:
    """Customizable business validation boundaries for client receipts."""

    min_company_length: int = 2
    max_company_length: int = 100
    min_address_length: int = 10
    max_address_length: int = 300
    min_total: float = 0.0
    max_total: float = 10000.0
    min_year: int = 2010
    max_year: int = 2030


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


@dataclass(frozen=True, slots=True)
class OcrFieldResult:
    """OCR result and operational status for one expected receipt field."""

    field: str
    text: str | None
    confidence: float | None
    status: str
    sharpness: float | None
    variant: str | None
    reason: str | None


@dataclass(slots=True)
class ReceiptValidationResult:
    """Consolidated business data ready for downstream consumption."""

    receipt_id: str
    company: str | None
    date: str | None
    address: str | None
    total: float | None
    validation_status: str
    errors: list[str]


# ----------------------------------------------------------------------
# YOLO ReceiptDetector
# ----------------------------------------------------------------------
class ReceiptDetector:
    """Runs receipt field localization on raw image paths using YOLOv8."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        if not config.model_path.is_file():
            raise ModelLoadError(f"YOLO weights file not found: {config.model_path}")

        try:
            import cv2
            from ultralytics import YOLO

            self._cv2 = cv2
            self.model = YOLO(str(config.model_path))
        except ImportError as error:
            raise ModelLoadError(
                "Missing packages (ultralytics or opencv-python). Please run: "
                "pip install ultralytics opencv-python"
            ) from error
        except Exception as error:
            raise ModelLoadError(f"Failed to load YOLO model: {error}") from error

    def load_image(self, path: str | Path) -> tuple[Path, Any]:
        """Read and validate the source image file."""
        resolved_path = Path(path).expanduser().resolve()
        if not resolved_path.is_file():
            raise ModelLoadError(f"Target receipt image not found: {resolved_path}")
        image = self._cv2.imread(str(resolved_path), self._cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ModelLoadError(f"OpenCV could not read target image: {resolved_path}")
        return resolved_path, image

    def predict(self, image_path: str | Path) -> tuple[Path, Any, list[Detection]]:
        """Run one model inference and apply unique class & overlap filtering."""
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
            raise ModelLoadError(f"YOLO inference failed for: {path}") from error

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

        # Enforce unique classes and suppress overlapping boxes
        accepted: list[Detection] = []
        seen_classes: set[str] = set()

        for candidate in detections:
            # 1. Enforce unique classes (keep highest confidence first)
            if candidate.class_name in seen_classes:
                self.logger.info(
                    "Filtered duplicate class '%s' with lower confidence %f in %s",
                    candidate.class_name,
                    candidate.confidence,
                    path.name,
                )
                continue

            # 2. Suppress overlaps between different classes (Intersection over Minimum Area > 0.20)
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


# ----------------------------------------------------------------------
# PaddleOCR ReceiptOcrProcessor
# ----------------------------------------------------------------------
class ReceiptOcrProcessor:
    """Reusable PaddleOCR engine running text recognition directly on in-memory crops."""

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        try:
            import cv2
            import numpy as np
            from paddleocr import PaddleOCR

            self._cv2 = cv2
            self._np = np
            self.engine = PaddleOCR(
                lang=config.language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        except ImportError as error:
            raise OcrProcessingError(
                "PaddleOCR/dependencies missing. Please run: "
                "pip install paddleocr opencv-python numpy"
            ) from error
        except Exception as error:
            raise OcrProcessingError(f"Unable to initialize PaddleOCR: {error}") from error

    def quality_metrics(self, image: Any) -> float:
        """Return Laplacian sharpness for a crop."""
        gray = self._cv2.cvtColor(image, self._cv2.IMREAD_GRAYSCALE or self._cv2.COLOR_BGR2GRAY)
        return float(self._cv2.Laplacian(gray, self._cv2.CV_64F).var())

    def variants(self, image: Any) -> Iterable[tuple[str, Any]]:
        """Yield conservative enhancements without replacing the original crop."""
        yield "original", image
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        clahe = self._cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = self._cv2.GaussianBlur(clahe, (0, 0), 2)
        sharpened = self._cv2.addWeighted(clahe, 1.5, blurred, -0.5, 0)
        yield "contrast_sharpened", self._cv2.cvtColor(sharpened, self._cv2.COLOR_GRAY2BGR)
        thresholded = self._cv2.adaptiveThreshold(
            clahe, 255, self._cv2.ADAPTIVE_THRESH_GAUSSIAN_C, self._cv2.THRESH_BINARY, 31, 9
        )
        yield "adaptive_threshold", self._cv2.cvtColor(thresholded, self._cv2.COLOR_GRAY2BGR)

    def _read_variant(self, image: Any) -> tuple[str, float] | None:
        """Extract text tokens and calculate average confidence."""
        result = list(self.engine.predict(image))
        texts: list[str] = []
        scores: list[float] = []
        self._collect_tokens(result, texts, scores)
        text = " ".join(item.strip() for item in texts if item.strip()).strip()
        return (text, sum(scores) / len(scores)) if text and scores else None

    def _collect_tokens(self, value: Any, texts: list[str], scores: list[float]) -> None:
        """Recursively parse text recognition tokens from PaddleOCR response."""
        if isinstance(value, str):
            try:
                self._collect_tokens(json.loads(value), texts, scores)
            except json.JSONDecodeError:
                return
            return
        if isinstance(value, dict):
            raw_text = value.get("rec_text") or value.get("text")
            raw_score = value.get("rec_score") or value.get("score")
            if isinstance(raw_text, str) and isinstance(raw_score, Real):
                texts.append(raw_text)
                scores.append(float(raw_score))
            for key in ("rec_texts", "rec_scores"):
                if key in value and isinstance(value[key], (list, tuple, self._np.ndarray)):
                    items = self._np.asarray(value[key]).tolist()
                    if key == "rec_texts":
                        texts.extend(str(item) for item in items)
                    else:
                        scores.extend(float(item) for item in items)
            for key, nested in value.items():
                if key not in {"rec_texts", "rec_scores"}:
                    self._collect_tokens(nested, texts, scores)
        elif isinstance(value, (list, tuple)):
            if (
                len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], Real)
            ):
                texts.append(value[0])
                scores.append(float(value[1]))
            else:
                for nested in value:
                    self._collect_tokens(nested, texts, scores)
        elif hasattr(value, "json"):
            payload = value.json
            resolved_payload = payload() if callable(payload) else payload
            self._collect_tokens(resolved_payload, texts, scores)
        elif isinstance(value, self._np.ndarray):
            self._collect_tokens(value.tolist(), texts, scores)

    def process_crop_in_memory(self, field: str, crop_image: Any) -> OcrFieldResult:
        """Run OCR on crop variants and apply quality gates."""
        if crop_image is None or crop_image.size == 0:
            return OcrFieldResult(
                field, None, None, "needs_manual_review", None, None, "unreadable_crop"
            )
        sharpness = self.quality_metrics(crop_image)
        candidates: list[tuple[str, str, float]] = []
        for name, variant in self.variants(crop_image):
            try:
                candidate = self._read_variant(variant)
            except Exception as error:
                self.logger.warning(
                    "OCR variant %s failed in memory: %s", name, error
                )
                continue
            if candidate:
                text, confidence = candidate
                candidates.append((name, text, confidence))
        if not candidates:
            return OcrFieldResult(
                field, None, None, "needs_manual_review", sharpness, None, "no_text_recognized"
            )
        variant, text, confidence = max(candidates, key=lambda item: item[2])
        if confidence < self.config.minimum_confidence:
            status, reason = "needs_manual_review", "low_ocr_confidence"
        elif sharpness < self.config.minimum_sharpness:
            status, reason = "needs_manual_review", "blurred_crop"
        else:
            status, reason = "accepted", None
        return OcrFieldResult(
            field,
            text,
            round(confidence, 6),
            status,
            round(sharpness, 3),
            variant,
            reason,
        )


# ----------------------------------------------------------------------
# ReceiptValidator
# ----------------------------------------------------------------------
class ReceiptValidator:
    """Validates raw OCR outputs against business logic rules."""

    def __init__(self, rules: BusinessRulesConfig) -> None:
        self.rules = rules

    def parse_date(self, raw_text: str | None) -> tuple[str | None, str | None]:
        """Normalize date to ISO 8601 YYYY-MM-DD. Returns (iso_date, error_message)."""
        if not raw_text:
            return None, "Date field is empty or missing."

        text = raw_text.strip()
        text = re.sub(r"^[^\w\d]+|[^\w\d]+$", "", text)
        if not text:
            return None, "Date field contains only noise."

        normalized = re.sub(r"[\-\.]", "/", text)

        # 1. Match DD/MM/YYYY or DD/MM/YY
        match_dmy = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", normalized)
        if match_dmy:
            day_str, month_str, year_str = match_dmy.groups()
            try:
                day, month, year = int(day_str), int(month_str), int(year_str)
                if year < 100:
                    year = 2000 + year
                if not (self.rules.min_year <= year <= self.rules.max_year):
                    return None, f"Year {year} is outside business bounds ({self.rules.min_year}-{self.rules.max_year})."
                dt = datetime.date(year, month, day)
                return dt.isoformat(), None
            except ValueError as error:
                return None, f"Invalid day/month values: {error}."

        # 2. Match YYYY/MM/DD
        match_ymd = re.search(r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b", normalized)
        if match_ymd:
            year_str, month_str, day_str = match_ymd.groups()
            try:
                year, month, day = int(year_str), int(month_str), int(day_str)
                if not (self.rules.min_year <= year <= self.rules.max_year):
                    return None, f"Year {year} is outside business bounds."
                dt = datetime.date(year, month, day)
                return dt.isoformat(), None
            except ValueError as error:
                return None, f"Invalid day/month values: {error}."

        # 3. Match named month (e.g. 12 Jan 2019)
        months_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
            "december": 12
        }
        match_word = re.search(r"\b(\d{1,2})[,\s]+([a-zA-Z]{3,})\b[,\s]+(\d{2,4})\b", text)
        if match_word:
            day_str, month_name, year_str = match_word.groups()
            month_name_lower = month_name.lower()
            if month_name_lower in months_map:
                try:
                    day, month, year = int(day_str), months_map[month_name_lower], int(year_str)
                    if year < 100:
                        year = 2000 + year
                    if not (self.rules.min_year <= year <= self.rules.max_year):
                        return None, f"Year {year} is outside business bounds."
                    dt = datetime.date(year, month, day)
                    return dt.isoformat(), None
                except ValueError as error:
                    return None, f"Invalid day/month values: {error}."

        # 4. Fallback search for digit combinations
        digits = re.sub(r"\D", "", text)
        if len(digits) == 8:
            try:
                # Try YYYYMMDD
                year, month, day = int(digits[0:4]), int(digits[4:6]), int(digits[6:8])
                if self.rules.min_year <= year <= self.rules.max_year:
                    dt = datetime.date(year, month, day)
                    return dt.isoformat(), None
            except ValueError:
                pass
            try:
                # Try DDMMYYYY
                day, month, year = int(digits[0:2]), int(digits[2:4]), int(digits[4:8])
                if self.rules.min_year <= year <= self.rules.max_year:
                    dt = datetime.date(year, month, day)
                    return dt.isoformat(), None
            except ValueError:
                pass
        elif len(digits) == 6:
            try:
                # Try DDMMYY
                day, month, year = int(digits[0:2]), int(digits[2:4]), 2000 + int(digits[4:6])
                if self.rules.min_year <= year <= self.rules.max_year:
                    dt = datetime.date(year, month, day)
                    return dt.isoformat(), None
            except ValueError:
                pass

        return None, f"Could not match '{raw_text}' against any known date formats."

    def parse_total(self, raw_text: str | None) -> tuple[float | None, str | None]:
        """Extract float value from total OCR text."""
        if not raw_text:
            return None, "Total field is empty or missing."

        text = raw_text.strip().upper()
        cleaned = re.sub(r"(?:RM|USD|SGD|MYR|\$|\bTOTAL\b|\bNET\b|\bAMT\b|\bAMOUNT\b)", "", text)
        cleaned = cleaned.replace(",", "").strip()

        match = re.search(r"\b\d+(?:\.\d{1,4})?\b", cleaned)
        if not match:
            match = re.search(r"\d+(?:\.\d+)?", cleaned)

        if not match:
            return None, f"Could not parse numeric total from '{raw_text}'."

        try:
            total_val = float(match.group(0))
            if not (self.rules.min_total <= total_val <= self.rules.max_total):
                return total_val, f"Total amount {total_val} is outside business bounds ({self.rules.min_total}-{self.rules.max_total})."
            return total_val, None
        except ValueError as error:
            return None, f"Value conversion error: {error}."

    def validate_receipt(self, raw_ocr_data: dict[str, any]) -> ReceiptValidationResult:
        """Run validation rules across all OCR receipt fields and aggregate errors."""
        receipt_id = raw_ocr_data.get("receipt_id", "unknown")
        fields_list = raw_ocr_data.get("fields", [])

        ocr_fields = {}
        ocr_statuses = {}
        for f in fields_list:
            field_name = f.get("field")
            if field_name:
                ocr_fields[field_name] = f.get("text")
                ocr_statuses[field_name] = {
                    "status": f.get("status"),
                    "reason": f.get("reason"),
                    "confidence": f.get("confidence"),
                }

        errors: list[str] = []

        # 1. Company Name Validation
        company_text = ocr_fields.get("company")
        validated_company = None
        if not company_text:
            errors.append("Field 'company' is missing or not detected.")
        else:
            cleaned_company = re.sub(r"\s+", " ", company_text).strip()
            if len(cleaned_company) < self.rules.min_company_length:
                errors.append(
                    f"Company name too short ({len(cleaned_company)} chars). Min: {self.rules.min_company_length}."
                )
            elif len(cleaned_company) > self.rules.max_company_length:
                errors.append(
                    f"Company name too long ({len(cleaned_company)} chars). Max: {self.rules.max_company_length}."
                )
            else:
                validated_company = cleaned_company

        # 2. Date Validation
        date_text = ocr_fields.get("date")
        validated_date, date_err = self.parse_date(date_text)
        if date_err:
            errors.append(f"Field 'date' validation error: {date_err}")

        # 3. Address Validation
        address_text = ocr_fields.get("address")
        validated_address = None
        if not address_text:
            errors.append("Field 'address' is missing or not detected.")
        else:
            cleaned_address = re.sub(r"\s+", " ", address_text).strip()
            if len(cleaned_address) < self.rules.min_address_length:
                errors.append(
                    f"Address too short ({len(cleaned_address)} chars). Min: {self.rules.min_address_length}."
                )
            elif len(cleaned_address) > self.rules.max_address_length:
                errors.append(
                    f"Address too long ({len(cleaned_address)} chars). Max: {self.rules.max_address_length}."
                )
            else:
                indicators = {
                    "NO", "LOT", "JALAN", "TAMAN", "ROAD", "STREET", "KUALA",
                    "SELANGOR", "JOHOR", "PENANG", "KPB"
                }
                words = {w.upper() for w in re.findall(r"\b[a-zA-Z]+\b", cleaned_address)}
                has_indicator = any(ind in words for ind in indicators)
                has_zip = bool(re.search(r"\b\d{5}\b", cleaned_address))
                if not (has_indicator or has_zip):
                    errors.append(
                        "Address does not contain typical postal indicators or zip code pattern."
                    )
                validated_address = cleaned_address

        # 4. Total Validation
        total_text = ocr_fields.get("total")
        validated_total, total_err = self.parse_total(total_text)
        if total_err:
            errors.append(f"Field 'total' validation error: {total_err}")

        # 5. Check OCR-level metadata flags
        for field_name, meta in ocr_statuses.items():
            ocr_status = meta.get("status")
            if ocr_status == "needs_manual_review":
                reason = meta.get("reason")
                conf = meta.get("confidence")
                errors.append(
                    f"Field '{field_name}' flagged by OCR: {reason} (Confidence: {conf})."
                )
            elif ocr_status == "not_detected":
                errors.append(f"Field '{field_name}' was not detected by YOLO model.")

        validation_status = "needs_manual_review" if errors else "valid"

        return ReceiptValidationResult(
            receipt_id=receipt_id,
            company=validated_company,
            date=validated_date,
            address=validated_address,
            total=validated_total,
            validation_status=validation_status,
            errors=errors,
        )


# ----------------------------------------------------------------------
# Visualization Drawing Helper
# ----------------------------------------------------------------------
def save_visualization(
    image: Any,
    source_stem: str,
    detections: Sequence[Detection],
    output_dir: Path = VISUALIZATION_DIR,
) -> Path:
    """Draw field labels on a copy of the original receipt and save it."""
    import cv2

    if image is None or image.size == 0:
        raise IRPSError("Cannot visualize detections on an empty image.")

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
        raise IRPSError(f"Unable to write visualization: {output_path}")
    return output_path


# ----------------------------------------------------------------------
# Pipeline Coordination (Processes in memory!)
# ----------------------------------------------------------------------
def process_single_image(
    image_path: Path,
    detector: ReceiptDetector,
    ocr_processor: ReceiptOcrProcessor,
    validator: ReceiptValidator,
    save_vis: bool = True,
    visualization_dir: Path = VISUALIZATION_DIR,
    json_dir: Path = JSON_DIR,
) -> dict[str, Any]:
    """Execute end-to-end YOLO detection, in-memory OCR, and validation for one image."""
    logger = get_logger("PipelineCoordinator")
    logger.info("Starting processing for image: %s", image_path.name)

    # 1. YOLO Detection
    image_path, image, detections = detector.predict(image_path)
    receipt_id = image_path.stem

    # 2. Visualization (Only folder 1 generated on disk if requested)
    if save_vis:
        save_visualization(image, receipt_id, detections, output_dir=visualization_dir)

    # 3. Crop and OCR in-memory (No folder/crops written to disk)
    detection_by_class = {det.class_name: det for det in detections}
    results = []
    for field in ["company", "date", "address", "total"]:
        if field not in detection_by_class:
            results.append(
                OcrFieldResult(
                    field,
                    None,
                    None,
                    "not_detected",
                    None,
                    None,
                    "detection_crop_missing",
                )
            )
        else:
            det = detection_by_class[field]
            crop_image = image[det.ymin : det.ymax, det.xmin : det.xmax]
            results.append(ocr_processor.process_crop_in_memory(field, crop_image))

    # 4. Construct OCR payload
    ocr_payload: dict[str, object] = {
        "receipt_id": receipt_id,
        "fields": [dataclasses.asdict(res) for res in results],
    }

    # 5. Validation and Normalization
    validation_result = validator.validate_receipt(ocr_payload)

    # 6. Save final output (Only folder 2 generated on disk)
    json_dir.mkdir(parents=True, exist_ok=True)
    dest_path = json_dir / f"{receipt_id}.json"

    payload = dataclasses.asdict(validation_result)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    logger.info(
        "Successfully finalized %s -> Status: %s. Saved to %s.",
        image_path.name,
        validation_result.validation_status,
        dest_path.name,
    )
    return payload


# ----------------------------------------------------------------------
# CLI Main Entry Point
# ----------------------------------------------------------------------
def main(arguments: Sequence[str] | None = None) -> int:
    """CLI orchestrator for the self-contained IRPS parsing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end YOLO + PaddleOCR + Validation pipeline."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single receipt image or a directory containing receipt images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to trained YOLO best.pt weights. Defaults to searching models/best.pt relative to the script.",
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
        "--minimum-confidence",
        type=float,
        default=0.75,
        help="PaddleOCR minimum confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run YOLO model on (cpu, cuda, 0, etc.).",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Do not save annotated detection visualization.",
    )

    parsed = parser.parse_args(arguments)
    logger = get_logger("PipelineMain")

    # Set up model path defaults
    yolo_model_path = parsed.model
    if not yolo_model_path:
        # Search relative to current script: scripts/../models/best.pt
        script_dir = Path(__file__).resolve().parent
        yolo_model_path = (script_dir.parent / "models" / "best.pt").resolve()

    yolo_cfg = InferenceConfig(
        model_path=yolo_model_path,
        confidence_threshold=parsed.confidence,
        iou_threshold=parsed.iou,
        device=parsed.device,
    )
    ocr_cfg = OcrConfig(minimum_confidence=parsed.minimum_confidence)
    rules_cfg = BusinessRulesConfig()

    try:
        logger.info("Initializing models...")
        detector = ReceiptDetector(yolo_cfg)
        ocr_processor = ReceiptOcrProcessor(ocr_cfg)
        validator = ReceiptValidator(rules_cfg)

        input_path = parsed.input_path.expanduser().resolve()

        if input_path.is_dir():
            image_paths = sorted(
                p
                for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
            )
            if not image_paths:
                logger.warning("No supported images found in directory: %s", input_path)
                return 0

            logger.info("Processing %d images in batch...", len(image_paths))
            success_count = 0
            for img_path in image_paths:
                try:
                    process_single_image(
                        img_path,
                        detector,
                        ocr_processor,
                        validator,
                        save_vis=not parsed.no_visualization,
                    )
                    success_count += 1
                except Exception as error:
                    logger.error("Failed processing %s: %s", img_path.name, error)

            print(
                json.dumps(
                    {
                        "processed_count": success_count,
                        "total_count": len(image_paths),
                        "message": "Pipeline processing completed.",
                    },
                    indent=2,
                )
            )
        else:
            # Single image processing
            result = process_single_image(
                input_path,
                detector,
                ocr_processor,
                validator,
                save_vis=not parsed.no_visualization,
            )
            print(json.dumps(result, indent=2))

        return 0
    except IRPSError as error:
        logger.error("Pipeline run failed: %s", error)
        return 1
    except Exception as error:
        logger.exception("Unexpected pipeline failure: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

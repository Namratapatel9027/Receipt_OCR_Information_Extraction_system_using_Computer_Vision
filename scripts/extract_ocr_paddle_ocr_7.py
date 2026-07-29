"""Self-contained PaddleOCR and business validation script.

Combines the OCR reading engine and JSON validation engine into a single script.
Reads crops from outputs/yolo_crops/ and outputs validated JSON to outputs/ocr_prediction/.
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

# Output Directory Configuration
OUTPUT_ROOT = Path("outputs")
OCR_PREDICTION_DIR = OUTPUT_ROOT / "ocr_prediction"

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


class ReceiptOcrProcessor:
    """Reusable PaddleOCR engine running text recognition on cropped files."""

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
            raise ImportError(
                "Missing packages (paddleocr, opencv-python, numpy). Please run: "
                "pip install paddleocr opencv-python numpy"
            ) from error
        except Exception as error:
            raise RuntimeError(f"Unable to initialize PaddleOCR: {error}") from error

    def quality_metrics(self, image: Any) -> float:
        """Return Laplacian sharpness for a crop."""
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
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

    def process_crop(self, field: str, crop_path: Path) -> OcrFieldResult:
        """Load crop and run PaddleOCR with quality gates."""
        image = self._cv2.imread(str(crop_path), self._cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return OcrFieldResult(
                field, None, None, "needs_manual_review", None, None, "unreadable_crop"
            )
        sharpness = self.quality_metrics(image)
        candidates: list[tuple[str, str, float]] = []
        for name, variant in self.variants(image):
            try:
                candidate = self._read_variant(variant)
            except Exception as error:
                self.logger.warning(
                    "OCR variant %s failed on disk: %s", name, error
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


def infer_field_name(path: Path) -> str | None:
    """Map crop filenames such as 01_total.jpg to a known field."""
    match = re.match(r"\d+_([a-z]+)$", path.stem.lower())
    return match.group(1) if match and match.group(1) in CLASS_MAPPING else None


def process_receipt(
    crop_dir: Path,
    ocr_config: OcrConfig,
    rules_config: BusinessRulesConfig,
    processor: ReceiptOcrProcessor | None = None,
) -> dict[str, Any]:
    """OCR all expected fields in crop_dir, run validator, and save output JSON."""
    if not crop_dir.is_dir():
        raise FileNotFoundError(f"Crop directory was not found: {crop_dir}")

    if processor is None:
        processor = ReceiptOcrProcessor(ocr_config)

    crops = sorted(
        path
        for path in crop_dir.iterdir()
        if path.suffix.lower() in VALID_IMAGE_SUFFIXES
    )
    by_field = {field: path for path in crops if (field := infer_field_name(path))}
    results = []
    for field in CLASS_MAPPING:
        if field not in by_field:
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
            results.append(processor.process_crop(field, by_field[field]))

    ocr_payload: dict[str, object] = {
        "receipt_id": crop_dir.name,
        "fields": [dataclasses.asdict(result) for result in results],
    }

    validator = ReceiptValidator(rules_config)
    validation_result = validator.validate_receipt(ocr_payload)

    OCR_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OCR_PREDICTION_DIR / f"{crop_dir.name}.json"
    payload = dataclasses.asdict(validation_result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return payload


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI orchestrator for OCR extraction and business validation."""
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR and business validation on receipt crops."
    )
    parser.add_argument(
        "crop_dir",
        type=Path,
        help="Path to single crop directory or parent crop directory containing multiple crops.",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.75,
        help="PaddleOCR minimum confidence threshold.",
    )
    parsed = parser.parse_args(arguments)
    logger = get_logger("OcrPipelineMain")

    ocr_cfg = OcrConfig(minimum_confidence=parsed.minimum_confidence)
    rules_cfg = BusinessRulesConfig()

    try:
        input_path = parsed.crop_dir.expanduser().resolve()

        if not input_path.is_dir():
            logger.error("Input crop directory does not exist: %s", input_path)
            return 1

        subdirs = sorted(p for p in input_path.iterdir() if p.is_dir())

        if subdirs:
            logger.info("Batch mode: Processing %d crop directories...", len(subdirs))
            processor = ReceiptOcrProcessor(ocr_cfg)
            success_count = 0
            for subdir in subdirs:
                try:
                    process_receipt(subdir, ocr_cfg, rules_cfg, processor)
                    success_count += 1
                except Exception as error:
                    logger.error("Failed processing %s: %s", subdir.name, error)
            print(
                json.dumps(
                    {
                        "processed_count": success_count,
                        "total_count": len(subdirs),
                        "message": "OCR and validation complete.",
                    },
                    indent=2,
                )
            )
        else:
            result = process_receipt(input_path, ocr_cfg, rules_cfg)
            print(json.dumps(result, indent=2))

        return 0
    except Exception as error:
        logger.exception("OCR run failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

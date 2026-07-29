import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# Ensure the root project directory is in the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app import models
from app.database import engine, get_db
from scripts.main_inference_8 import (
    BusinessRulesConfig,
    InferenceConfig,
    OcrConfig,
    OcrFieldResult,
    ReceiptDetector,
    ReceiptOcrProcessor,
    ReceiptValidator,
    save_visualization,
)

# Create the SQLite database tables on startup if they don't exist
models.Base.metadata.create_all(bind=engine)

# Define directories
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "yolo_visualization"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

JSON_DIR = PROJECT_ROOT / "outputs" / "ocr_final_json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="IRPS API & Portal",
    description="Backend API and Web Portal for Intelligent Receipt Processing System",
    version="1.0.0",
)

# Serve the visualizer images publicly so the frontend can load them
app.mount(
    "/static/visualizations",
    StaticFiles(directory=str(VISUALIZATION_DIR)),
    name="visualizations",
)

# ----------------------------------------------------------------------
# Model Initialization (Loaded once at startup in memory)
# ----------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
print(f"Loading YOLO model from: {MODEL_PATH}...")

detector = ReceiptDetector(
    InferenceConfig(model_path=MODEL_PATH, device="cpu", verbose=False)
)
ocr_processor = ReceiptOcrProcessor(OcrConfig())
validator = ReceiptValidator(BusinessRulesConfig())

print("Models loaded successfully! Server is ready.")


# ----------------------------------------------------------------------
# API Routes
# ----------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def serve_portal():
    """Serve the premium frontend landing page."""
    template_path = PROJECT_ROOT / "app" / "templates" / "index.html"
    if not template_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Portal template file index.html not found.",
        )
    with template_path.open("r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
def predict_receipt(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload receipt, run YOLO + OCR, persist in database, and return JSON."""
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_suffix}'. Please upload a JPG, PNG, or WEBP image.",
        )

    # 1. Save uploaded file
    local_image_path = UPLOADS_DIR / file.filename
    try:
        with local_image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {error}"
        )

    # 2. Run local ML pipeline steps manually to extract confidence scores
    try:
        # Run YOLO
        image_path, image, detections = detector.predict(local_image_path)
        receipt_id = local_image_path.stem

        # Save visualization with boxes
        save_visualization(image, receipt_id, detections, output_dir=VISUALIZATION_DIR)

        # Crop and OCR in-memory
        detection_by_class = {det.class_name: det for det in detections}
        ocr_results = []
        for field in ["company", "date", "address", "total"]:
            if field not in detection_by_class:
                ocr_results.append(
                    OcrFieldResult(
                        field=field,
                        text=None,
                        confidence=None,
                        status="not_detected",
                        sharpness=None,
                        variant=None,
                        reason="detection_crop_missing",
                    )
                )
            else:
                det = detection_by_class[field]
                crop_image = image[det.ymin : det.ymax, det.xmin : det.xmax]
                ocr_results.append(
                    ocr_processor.process_crop_in_memory(field, crop_image)
                )

        # Build OCR payload for validation
        ocr_payload = {
            "receipt_id": receipt_id,
            "fields": [dataclasses.asdict(res) for res in ocr_results],
        }

        # Validate
        validation_result = validator.validate_receipt(ocr_payload)
        payload = dataclasses.asdict(validation_result)

        # Calculate average confidence of successfully read OCR fields
        conf_scores = [
            res.confidence for res in ocr_results if res.confidence is not None
        ]
        avg_confidence = (
            sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
        )
        payload["confidence"] = round(avg_confidence, 4)

        # Save final OCR JSON to disk
        dest_path = JSON_DIR / f"{receipt_id}.json"
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Inference pipeline failed: {error}"
        )

    # 3. Write database record
    errors_string = ",".join(payload["errors"]) if payload["errors"] else ""
    existing_record = (
        db.query(models.Receipt)
        .filter(models.Receipt.receipt_id == receipt_id)
        .first()
    )

    if existing_record:
        # Update existing
        existing_record.company = payload["company"]
        existing_record.date = payload["date"]
        existing_record.address = payload["address"]
        existing_record.total = payload["total"]
        existing_record.validation_status = payload["validation_status"]
        existing_record.errors = errors_string
        db.commit()
        db.refresh(existing_record)
        db_receipt = existing_record
    else:
        # Insert new
        db_receipt = models.Receipt(
            receipt_id=receipt_id,
            company=payload["company"],
            date=payload["date"],
            address=payload["address"],
            total=payload["total"],
            validation_status=payload["validation_status"],
            errors=errors_string,
        )
        db.add(db_receipt)
        db.commit()
        db.refresh(db_receipt)

    return {
        "message": "Processing completed successfully.",
        "db_record_id": db_receipt.id,
        "data": payload,
        "visualization_url": f"/static/visualizations/{receipt_id}_detections.jpg",
    }


@app.get("/receipts")
def list_receipts(db: Session = Depends(get_db)):
    """Fetch all processed receipts currently stored in SQLite."""
    receipts = db.query(models.Receipt).order_by(models.Receipt.id.desc()).all()
    return receipts


@app.get("/receipts/{receipt_id}")
def get_receipt(receipt_id: str, db: Session = Depends(get_db)):
    """Retrieve details of a single receipt by its receipt_id."""
    receipt = (
        db.query(models.Receipt)
        .filter(models.Receipt.receipt_id == receipt_id)
        .first()
    )
    if not receipt:
        raise HTTPException(
            status_code=404,
            detail=f"Receipt with ID '{receipt_id}' not found in the database.",
        )
    return receipt

import os
import shutil
import sys
from pathlib import Path
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

# Ensure the root project directory is in the Python path so we can import scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app import models
from app.database import engine, get_db
from scripts.main_inference_8 import (
    BusinessRulesConfig,
    InferenceConfig,
    OcrConfig,
    ReceiptDetector,
    ReceiptOcrProcessor,
    ReceiptValidator,
    process_single_image,
)

# Create the SQLite database tables on startup if they don't exist
models.Base.metadata.create_all(bind=engine)

# Define file storage paths
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the FastAPI App
app = FastAPI(
    title="Intelligent Receipt Processing System (IRPS) API",
    description="Backend API for YOLOv8 Field Localization and PaddleOCR Text Extraction",
    version="1.0.0",
)

# ----------------------------------------------------------------------
# Model Initialization (Loaded once at startup in memory)
# ----------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
print(f"Loading YOLO model from: {MODEL_PATH}...")

# Load models globally so they don't reload on each request (which would be very slow)
detector = ReceiptDetector(
    InferenceConfig(model_path=MODEL_PATH, device="cpu", verbose=False)
)
ocr_processor = ReceiptOcrProcessor(OcrConfig())
validator = ReceiptValidator(BusinessRulesConfig())

print("Models loaded successfully! Server is ready.")


# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------


@app.get("/")
def read_root():
    """Welcome endpoint confirming server status."""
    return {
        "status": "online",
        "message": "Welcome to the IRPS API. Go to /docs for the swagger documentation.",
    }


@app.post("/predict")
def predict_receipt(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a receipt image, run YOLO and PaddleOCR extraction, save to SQLite, and return JSON."""
    # 1. Validate file extension
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_suffix}'. Please upload a JPG, PNG, or WEBP image.",
        )

    # 2. Save the uploaded file locally to data/uploads/
    local_image_path = UPLOADS_DIR / file.filename
    try:
        with local_image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {error}"
        )

    # 3. Run the ML inference pipeline in-memory
    try:
        payload = process_single_image(
            image_path=local_image_path,
            detector=detector,
            ocr_processor=ocr_processor,
            validator=validator,
            save_vis=True,
            visualization_dir=PROJECT_ROOT / "outputs" / "yolo_visualization",
            json_dir=PROJECT_ROOT / "outputs" / "ocr_final_json",
        )
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Inference pipeline failed: {error}"
        )

    # 4. Check if a record with the same receipt_id already exists in the database
    receipt_id = payload["receipt_id"]
    existing_record = (
        db.query(models.Receipt)
        .filter(models.Receipt.receipt_id == receipt_id)
        .first()
    )

    errors_string = ",".join(payload["errors"]) if payload["errors"] else ""

    if existing_record:
        # Update existing record
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
        # Create new database record
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

    # 5. Return the extracted and validated receipt data
    return {
        "message": "Processing completed successfully.",
        "db_record_id": db_receipt.id,
        "data": payload,
    }


@app.get("/receipts")
def list_receipts(db: Session = Depends(get_db)):
    """Fetch all processed receipts currently stored in the SQLite database."""
    receipts = db.query(models.Receipt).all()
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

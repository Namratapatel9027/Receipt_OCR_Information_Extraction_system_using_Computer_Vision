import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String
from app.database import Base


class Receipt(Base):
    """Database table model for storing extracted and validated receipt data."""

    __tablename__ = "receipts"

    # Unique database ID (primary key)
    id = Column(Integer, primary_key=True, index=True)

    # Name of the file/receipt stem (e.g. X51005568885)
    receipt_id = Column(String, unique=True, index=True, nullable=False)

    # Extracted fields
    company = Column(String, nullable=True)
    date = Column(String, nullable=True)
    address = Column(String, nullable=True)
    total = Column(Float, nullable=True)

    # Validation metadata
    validation_status = Column(String, nullable=False)  # "valid" or "needs_manual_review"
    errors = Column(String, nullable=True)  # Store errors as a comma-separated string

    # Timestamp of when the receipt was processed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

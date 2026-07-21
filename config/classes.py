"""
Pixonate Labs Pvt. Ltd.
IRPS - Intelligent Receipt Processing System
Configuration file defining fixed class mappings for the detection pipeline.
"""

from typing import Dict, List

# Fixed class mapping as required by production contract
CLASS_MAPPING: Dict[str, int] = {
    "company": 0,
    "date": 1,
    "address": 2,
    "total": 3
}

# Inverse mapping for visualization and inference rendering
REV_CLASS_MAPPING: Dict[int, str] = {v: k for k, v in CLASS_MAPPING.items()}

# Strict production validation list
VALID_CLASSES: List[str] = list(CLASS_MAPPING.keys())

def get_class_index(class_name: str) -> int:
    """Returns the integer index for a given class name string with validation."""
    normalized_name = class_name.strip().lower()
    if normalized_name not in CLASS_MAPPING:
        raise ValueError(f"Invalid class name requested: {class_name}. Must be one of {VALID_CLASSES}")
    return CLASS_MAPPING[normalized_name]
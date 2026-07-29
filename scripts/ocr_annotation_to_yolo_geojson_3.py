"""
Pixonate Labs Pvt. Ltd.
IRPS - Intelligent Receipt Processing System
Unified high-precision annotation conversion pipeline producing QuPath-compliant
GeoJSONs vector layers for manual quality control annotation adjustments.
"""

import os
import re
import json
import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IRPS_GeoJsonPipeline")

CLASS_MAPPING = {"company": 0, "date": 1, "address": 2, "total": 3}
CLASS_COLORS = {
    0: (0, 0, 255),    # Company -> Red
    1: (0, 255, 0),    # Date -> Green
    2: (255, 0, 0),    # Address -> Blue
    3: (0, 255, 255)   # Total -> Yellow
}

# QuPath decimal color values mapped for immediate visual recognition on import
QUPATH_COLORS = {
    0: -65536,       # Red
    1: -16711936,    # Green
    2: -16776961,    # Blue
    3: -256          # Yellow
}

def clean_text(text: str) -> str:
    """Standardizes text structures for sequence character checks."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def find_dense_cluster(boxes: List[List[int]], line_height_thresh: float = 2.5) -> List[List[int]]:
    """Groups matched components using text block spatial height metrics to avoid bleeding."""
    if not boxes:
        return []
        
    sorted_boxes = sorted(boxes, key=lambda b: min(b[1::2]))
    heights = [max(b[1::2]) - min(b[1::2]) for b in sorted_boxes]
    median_height = np.median(heights) if heights else 15.0
    max_gap = median_height * line_height_thresh
    
    clusters: List[List[List[int]]] = []
    current_cluster: List[List[int]] = [sorted_boxes[0]]
    
    for next_box in sorted_boxes[1:]:
        prev_box_bottom = max(current_cluster[-1][1::2])
        next_box_top = min(next_box[1::2])
        
        if (next_box_top - prev_box_bottom) <= max_gap:
            current_cluster.append(next_box)
        else:
            clusters.append(current_cluster)
            current_cluster = [next_box]
    clusters.append(current_cluster)
    
    return max(clusters, key=len)

def create_qupath_feature(bbox: List[int], field_name: str, class_id: int) -> Dict[str, Any]:
    """Generates a closed-loop GeoJSON polygon matching QuPath's internal feature standard."""
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # QuPath requires coordinate array loops to be closed explicitly (first point == last point)
    polygon_coordinates = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin]
    ]
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon_coordinates]
        },
        "properties": {
            "name": field_name,
            "classification": {
                "name": field_name,
                "colorRGB": QUPATH_COLORS.get(class_id, -1)
            }
        }
    }

def process_receipt(base_id: str, img_path: str, box_path: str, ent_path: str, 
                    out_geo_dir: str, debug_dir: str, ocr_debug_dir: str):
    """Processes a single receipt layout, mapping text via spatial density filters into GeoJSON files."""
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w, _ = img.shape

    with open(ent_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
        
    ocr_lines = []
    with open(box_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(',', 8)
            if len(parts) >= 9:
                try:
                    coords = [int(p) for p in parts[:8]]
                    text = parts[8].strip()
                    ocr_lines.append((coords, text, clean_text(text)))
                except ValueError:
                    continue

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Save Raw OCR Tokens Image
    # -------------------------------------------------------------
    ocr_canvas = img.copy()
    for coords, raw_text, _ in ocr_lines:
        xs = coords[0::2]
        ys = coords[1::2]
        cv2.rectangle(ocr_canvas, (min(xs), min(ys)), (max(xs), max(ys)), (200, 200, 200), 1)
    cv2.imwrite(os.path.join(ocr_debug_dir, f"{base_id}_raw_ocr.jpg"), ocr_canvas)

    # -------------------------------------------------------------
    # MAPPING ENGINE: Core Spatial Filtering & Feature Generation
    # -------------------------------------------------------------
    geojson_features = []
    final_canvas = img.copy()

    for field, target_val in entities.items():
        if field not in CLASS_MAPPING:
            continue
        class_id = CLASS_MAPPING[field]
        cleaned_target = clean_text(target_val)
        if not cleaned_target:
            continue

        candidate_boxes = []
        for coords, _, clean_txt in ocr_lines:
            if not clean_txt:
                continue
            if clean_txt in cleaned_target or cleaned_target in clean_txt:
                candidate_boxes.append(coords)

        matched_boxes = find_dense_cluster(candidate_boxes)

        if matched_boxes:
            all_xs = [x for box in matched_boxes for x in box[0::2]]
            all_ys = [y for box in matched_boxes for y in box[1::2]]
            bbox = [min(all_xs), min(all_ys), max(all_xs), max(all_ys)]

            # Generate QuPath GeoJSON component object
            feature = create_qupath_feature(bbox, field, class_id)
            geojson_features.append(feature)

            # Draw validation preview box
            color = CLASS_COLORS[class_id]
            cv2.rectangle(final_canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(final_canvas, field, (bbox[0], max(bbox[1] - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save output GeoJSON feature collection file
    qupath_geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }
    
    geo_out_path = os.path.join(out_geo_dir, f"{base_id}.geojson")
    with open(geo_out_path, 'w', encoding='utf-8') as f:
        json.dump(qupath_geojson, f, indent=2)
        
    cv2.imwrite(os.path.join(debug_dir, f"{base_id}_final_preview.jpg"), final_canvas)

def main():
    for split in ["train", "test"]:
        img_dir, box_dir, ent_dir = f"data/raw/{split}/img", f"data/raw/{split}/box", f"data/raw/{split}/entities"
        out_geo_dir, debug_dir = f"data/processed/geojson/{split}", f"data/processed/debug/{split}"
        ocr_debug_dir = f"data/processed/ocr_debug/{split}"
        
        os.makedirs(out_geo_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(ocr_debug_dir, exist_ok=True)
        
        if not os.path.exists(img_dir):
            continue
            
        logger.info(f"Running pipeline for split: {split}")
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            base_id = os.path.splitext(img_name)[0]
            process_receipt(base_id, os.path.join(img_dir, img_name), os.path.join(box_dir, f"{base_id}.txt"),
                            os.path.join(ent_dir, f"{base_id}.txt"), out_geo_dir, debug_dir, ocr_debug_dir)
    logger.info("GeoJSON production vectors successfully built and saved.")

if __name__ == "__main__":
    main()
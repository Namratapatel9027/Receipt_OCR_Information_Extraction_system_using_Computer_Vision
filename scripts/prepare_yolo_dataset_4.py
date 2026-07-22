"""
Pixonate Labs Pvt. Ltd.
IRPS - Intelligent Receipt Processing System
Production-grade generic dataset parser: Splits annotations into strict 80/20 
train/val distributions, calculates normalized YOLO matrices, and outputs diagnostic layers.
"""

import os
import json
import logging
import random
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

# ----------------------------------------------------------------------
# 1. Pipeline Logging Engine Configuration
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IRPS_DatasetSplitter")

# ----------------------------------------------------------------------
# 2. Production Parameter Configuration Context
# ----------------------------------------------------------------------
class PipelineConfig:
    # Target Inputs (Modify these fields to repurpose for any dataset)
    SOURCE_GEOJSON_DIR = "/home/khushi/Codebasics/Dataset/SROIE2019/qu/geojsons_50"
    SOURCE_IMAGE_DIR = "/home/khushi/Codebasics/Dataset/SROIE2019/IRPS/data/raw/train/img"
    
    # Target Production Outputs
    OUTPUT_BASE_DIR = "data/processed/yolo_1"
    
    # Structural Machine Learning Rules
    TRAIN_SPLIT_RATIO = 0.80  # Strict 80% Training Allocation
    RANDOM_SEED = 42          # Locked seed for exact split reproducibility
    
    # Operational File Constraints
    VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
    
    # Fixed Class Index Mapping Strategy
    CLASS_MAPPING: Dict[str, int] = {
        "company": 0,
        "date": 1,
        "address": 2,
        "total": 3
    }
    
    # Diagnostic Visualization Color Maps (BGR Format)
    CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
        0: (0, 0, 255),    # Company -> Red
        1: (0, 255, 0),    # Date -> Green
        2: (255, 0, 0),    # Address -> Blue
        3: (0, 255, 255)   # Total -> Yellow
    }

# ----------------------------------------------------------------------
# 3. Core Processing Functional Modules
# ----------------------------------------------------------------------
def parse_geojson_geometry(geojson_path: str, class_map: Dict[str, int]) -> List[Tuple[int, int, int, int, int]]:
    """Reads a closed-loop QuPath polygon and unrolls absolute box dimensions."""
    extracted_boxes = []
    if not os.path.exists(geojson_path):
        return extracted_boxes

    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        features = data.get("features", [])
        for feature in features:
            props = feature.get("properties", {})
            class_name = props.get("name", "").lower().strip()
            
            if class_name not in class_map:
                continue
            class_id = class_map[class_name]
            
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon":
                coordinates = geom.get("coordinates", [[]])[0]
                if len(coordinates) >= 4:
                    xs = [int(point[0]) for point in coordinates]
                    ys = [int(point[1]) for point in coordinates]
                    extracted_boxes.append((class_id, min(xs), min(ys), max(xs), max(ys)))
    except Exception as e:
        logger.error(f"Failed parsing vector boundaries inside {geojson_path}: {str(e)}")
        
    return extracted_boxes

def normalize_to_yolo_format(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Applies coordinate fraction normalization with clipping guards."""
    xmin, ymin, xmax, ymax = bbox
    
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + (box_width / 2.0)
    y_center = ymin + (box_height / 2.0)
    
    nx = float(np.clip(x_center / img_w, 0.0, 1.0))
    ny = float(np.clip(y_center / img_h, 0.0, 1.0))
    nw = float(np.clip(box_width / img_w, 0.0, 1.0))
    nh = float(np.clip(box_height / img_h, 0.0, 1.0))
    
    return nx, ny, nw, nh

def write_yolo_meta_yaml(output_file: str, base_root_dir: str, class_map: Dict[str, int]):
    """Generates the deployment descriptor data.yaml file for YOLOv8 model imports."""
    reversed_names = {v: k for k, v in class_map.items()}
    names_block = "\n".join([f"  {idx}: {name}" for idx, name in sorted(reversed_names.items())])
    
    abs_data_root = os.path.abspath(base_root_dir)
    yaml_content = f"""path: {abs_data_root}
train: images/train
val: images/val
test: images/val

names:
{names_block}
"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    logger.info(f"Production training deployment data.yaml successfully written to: {output_file}")

# ----------------------------------------------------------------------
# 4. Main Execution Pipeline Coordinator
# ----------------------------------------------------------------------
def main():
    logger.info("Starting production dataset partition and normalization pipeline...")
    
    # 1. Verify existence of primary data entry channels
    if not os.path.exists(PipelineConfig.SOURCE_GEOJSON_DIR):
        logger.error(f"Source GeoJSON folder path not found: {PipelineConfig.SOURCE_GEOJSON_DIR}")
        return
    if not os.path.exists(PipelineConfig.SOURCE_IMAGE_DIR):
        logger.error(f"Source Image folder path not found: {PipelineConfig.SOURCE_IMAGE_DIR}")
        return

    # 2. Isolate all matching files that contain both image and annotation layers
    all_geojson_files = [f for f in os.listdir(PipelineConfig.SOURCE_GEOJSON_DIR) if f.endswith('.geojson')]
    valid_sample_ids = []
    
    for geo_name in all_geojson_files:
        base_id = os.path.splitext(geo_name)[0]
        # Confirm matching source image exists
        image_exists = False
        for ext in PipelineConfig.VALID_EXTENSIONS:
            if os.path.exists(os.path.join(PipelineConfig.SOURCE_IMAGE_DIR, f"{base_id}{ext}")):
                image_exists = True
                break
        if image_exists:
            valid_sample_ids.append(base_id)
            
    total_count = len(valid_sample_ids)
    logger.info(f"Total matching production validation samples identified: {total_count}")
    if total_count == 0:
        logger.error("Zero matching image and geojson structural pairs discovered. Check file mappings.")
        return

    # 3. Apply strict, seed-locked randomized distribution splitting
    random.seed(PipelineConfig.RANDOM_SEED)
    random.shuffle(valid_sample_ids)
    
    split_index = int(total_count * PipelineConfig.TRAIN_SPLIT_RATIO)
    train_ids = valid_sample_ids[:split_index]
    val_ids = valid_sample_ids[split_index:]
    
    split_assignment = {
        "train": train_ids,
        "val": val_ids
    }
    
    logger.info(f"Data partitioning structure computed: Train = {len(train_ids)} files | Val = {len(val_ids)} files")

    # 4. Process splits and compile outputs into target locations
    for split_name, id_list in split_assignment.items():
        img_out_dir = os.path.join(PipelineConfig.OUTPUT_BASE_DIR, "images", split_name)
        lbl_out_dir = os.path.join(PipelineConfig.OUTPUT_BASE_DIR, "labels", split_name)
        dbg_out_dir = os.path.join(PipelineConfig.OUTPUT_BASE_DIR, "prep_debug", split_name)
        
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        os.makedirs(dbg_out_dir, exist_ok=True)
        
        for sample_id in id_list:
            # Reconstruct absolute paths
            geojson_path = os.path.join(PipelineConfig.SOURCE_GEOJSON_DIR, f"{sample_id}.geojson")
            
            matched_img_file = None
            for ext in PipelineConfig.VALID_EXTENSIONS:
                test_path = os.path.join(PipelineConfig.SOURCE_IMAGE_DIR, f"{sample_id}{ext}")
                if os.path.exists(test_path):
                    matched_img_file = f"{sample_id}{ext}"
                    break
            
            src_img_path = os.path.join(PipelineConfig.SOURCE_IMAGE_DIR, matched_img_file)
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            # Read QuPath vector features
            raw_boxes = parse_geojson_geometry(geojson_path, PipelineConfig.CLASS_MAPPING)
            yolo_lines = []
            canvas = img.copy()
            
            for class_id, xmin, ymin, xmax, ymax in raw_boxes:
                # Compile normalized parameters
                nx, ny, nw, nh = normalize_to_yolo_format((xmin, ymin, xmax, ymax), w, h)
                yolo_lines.append(f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                
                # Render diagnostic color check box
                color = PipelineConfig.CLASS_COLORS.get(class_id, (0, 255, 0))
                cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(canvas, f"Class_{class_id}", (xmin, max(ymin - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
            # Save standard image asset to processed directory split
            cv2.imwrite(os.path.join(img_out_dir, matched_img_file), img)
            
            # Save normalized matrix text file
            with open(os.path.join(lbl_out_dir, f"{sample_id}.txt"), 'w', encoding='utf-8') as f:
                f.writelines(yolo_lines)
                
            # Save image diagnostic frame check
            cv2.imwrite(os.path.join(dbg_out_dir, f"{sample_id}_check.jpg"), canvas)
            
        logger.info(f"Split distribution processing complete for channel: '{split_name}'")

    # 5. Build central deployment roadmap configuration mapping file
    write_yolo_meta_yaml(
        output_file=os.path.join(PipelineConfig.OUTPUT_BASE_DIR, "data.yaml"),
        base_root_dir=PipelineConfig.OUTPUT_BASE_DIR,
        class_map=PipelineConfig.CLASS_MAPPING
    )
    logger.info("YOLO training matrix compilation and splitting phase successfully finalized.")

if __name__ == "__main__":
    main()
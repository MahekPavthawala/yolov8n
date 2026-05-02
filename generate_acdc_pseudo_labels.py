# -*- coding: utf-8 -*-
"""
Generate pseudo-labels for ACDC lane segmentation using a BDD100K-trained YOLOv8n-seg model.

Input
-----
- Trained lane model from BDD100K
- ACDC train images

Output
------
acdc_lane_pseudo/
├── images/train
├── labels/train
├── acdc_lane_pseudo.yaml
├── pseudo_label_summary.json
└── pseudo_label_per_image.csv

Notes
-----
- Assumes single-class lane segmentation
- Keeps only confident masks
- Writes YOLO segmentation label format:
    class x1 y1 x2 y2 ... xn yn
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================
MODEL_PT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
    r"\yolov8n_bdd100k_lane_seg2\weights\best.pt"
)

ACDC_IMAGE_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\ACDC\rgb_anon\rgb_anon"
)

OUTPUT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_lane_pseudo"
)
# ============================================================


# ============================================================
# SETTINGS
# ============================================================
WEATHERS = ["fog", "night", "rain", "snow"]
SPLIT_NAME = "train"

IMGSZ = 640
DEVICE = 0
CONF = 0.35
IOU = 0.45

# polygon filtering
MIN_POLYGON_POINTS = 6       # minimum number of contour points to keep
MIN_AREA_PX = 40             # remove tiny masks
CLASS_ID = 0                 # single-class lane segmentation
# ============================================================


def ensure_paths():
    required = [MODEL_PT, ACDC_IMAGE_ROOT]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path:\n{p}")


def prepare_dirs():
    (OUTPUT_ROOT / "images" / SPLIT_NAME).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / SPLIT_NAME).mkdir(parents=True, exist_ok=True)


def flatten_rel_path(rel_path: Path) -> str:
    return "_".join(rel_path.parts)


def find_weather_images(weather: str):
    weather_dir = ACDC_IMAGE_ROOT / weather / SPLIT_NAME
    if not weather_dir.exists():
        print(f"[WARNING] Missing weather folder: {weather_dir}")
        return []

    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(weather_dir.rglob(ext))
    return sorted(image_files)


def mask_to_polygons(mask_uint8: np.ndarray):
    """
    Convert binary mask to YOLO segmentation polygons.
    Returns list of normalized polygon strings (without class id).
    """
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask_uint8.shape[:2]
    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX:
            continue

        cnt = cnt.squeeze(axis=1) if len(cnt.shape) == 3 else cnt
        if cnt.ndim != 2 or cnt.shape[0] < MIN_POLYGON_POINTS:
            continue

        pts = []
        for x, y in cnt:
            xn = min(max(float(x) / w, 0.0), 1.0)
            yn = min(max(float(y) / h, 0.0), 1.0)
            pts.extend([xn, yn])

        # need at least 3 points => 6 numbers
        if len(pts) >= 2 * MIN_POLYGON_POINTS:
            polygons.append(pts)

    return polygons


def write_yaml():
    yaml_path = OUTPUT_ROOT / "acdc_lane_pseudo.yaml"
    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/train

names:
  0: lane
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\nYAML written to:\n{yaml_path}")


def main():
    ensure_paths()
    prepare_dirs()

    model = YOLO(str(MODEL_PT))

    out_img_dir = OUTPUT_ROOT / "images" / SPLIT_NAME
    out_lbl_dir = OUTPUT_ROOT / "labels" / SPLIT_NAME

    per_image_rows = []

    total_images = 0
    images_with_masks = 0
    empty_label_images = 0
    total_masks_kept = 0
    conf_values = []

    for weather in WEATHERS:
        print(f"\n================ {weather.upper()} =================\n")
        image_files = find_weather_images(weather)
        print(f"Images found: {len(image_files)}")

        for idx, img_path in enumerate(image_files, start=1):
            rel_path = img_path.relative_to(ACDC_IMAGE_ROOT)
            dst_name = flatten_rel_path(rel_path)
            dst_img_path = out_img_dir / dst_name
            dst_lbl_path = out_lbl_dir / f"{Path(dst_name).stem}.txt"

            # copy image into pseudo dataset
            if not dst_img_path.exists():
                shutil.copy2(img_path, dst_img_path)

            total_images += 1

            results = model.predict(
                source=str(img_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False,
                save=False
            )

            label_lines = []
            image_mask_count = 0
            image_conf_list = []

            if len(results) > 0:
                r = results[0]

                if r.masks is not None and r.boxes is not None:
                    masks = r.masks.data.cpu().numpy()      # [N, H, W]
                    confs = r.boxes.conf.cpu().numpy()      # [N]

                    for mask_arr, conf_score in zip(masks, confs):
                        mask_bin = (mask_arr > 0.5).astype(np.uint8) * 255
                        polygons = mask_to_polygons(mask_bin)

                        for poly in polygons:
                            poly_str = " ".join(f"{v:.6f}" for v in poly)
                            label_lines.append(f"{CLASS_ID} {poly_str}")
                            image_mask_count += 1
                            image_conf_list.append(float(conf_score))
                            conf_values.append(float(conf_score))

            dst_lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

            if image_mask_count > 0:
                images_with_masks += 1
                total_masks_kept += image_mask_count
            else:
                empty_label_images += 1

            per_image_rows.append({
                "weather": weather,
                "image_name": dst_name,
                "source_path": str(img_path),
                "masks_kept": image_mask_count,
                "avg_conf": float(np.mean(image_conf_list)) if image_conf_list else 0.0,
                "has_detection": int(image_mask_count > 0),
            })

            if idx % 50 == 0 or idx == len(image_files):
                print(f"Processed {idx}/{len(image_files)}")

    write_yaml()

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_csv = OUTPUT_ROOT / "pseudo_label_per_image.csv"
    per_image_df.to_csv(per_image_csv, index=False)

    summary = {
        "model_checkpoint": str(MODEL_PT),
        "source_root": str(ACDC_IMAGE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "settings": {
            "imgsz": IMGSZ,
            "device": DEVICE,
            "conf": CONF,
            "iou": IOU,
            "min_polygon_points": MIN_POLYGON_POINTS,
            "min_area_px": MIN_AREA_PX,
        },
        "totals": {
            "images_processed": total_images,
            "images_with_masks": images_with_masks,
            "empty_label_images": empty_label_images,
            "total_masks_kept": total_masks_kept,
            "avg_conf_all_masks": float(np.mean(conf_values)) if conf_values else 0.0,
        },
        "weather_summary": (
            per_image_df.groupby("weather")
            .agg(
                images=("image_name", "count"),
                images_with_detection=("has_detection", "sum"),
                total_masks=("masks_kept", "sum"),
                avg_masks_per_image=("masks_kept", "mean"),
                avg_conf=("avg_conf", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    }

    summary_path = OUTPUT_ROOT / "pseudo_label_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ PSEUDO-LABEL SUMMARY ================\n")
    print(f"Images processed     : {total_images}")
    print(f"Images with masks    : {images_with_masks}")
    print(f"Empty label images   : {empty_label_images}")
    print(f"Total masks kept     : {total_masks_kept}")
    print(f"Average mask conf    : {summary['totals']['avg_conf_all_masks']:.4f}")
    print(f"\nPer-image CSV        : {per_image_csv}")
    print(f"Summary JSON         : {summary_path}")
    print(f"Dataset root         : {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Evaluate adapted YOLOv8n-seg lane model on ACDC weather splits
and save overlay prediction images.

Outputs
-------
- per-image CSV
- weather summary CSV
- weather summary JSON
- saved prediction overlay images with segmented lanes

Author: mahek
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================
MODEL_PT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
    r"\yolov8n_bdd_lane_to_acdc_pseudo\weights\best.pt"
)

ACDC_IMAGE_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\ACDC\rgb_anon\rgb_anon"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
)

RUN_NAME = "yolov8n_bdd_lane_to_acdc_pseudo_eval"
# ============================================================


# ============================================================
# SETTINGS
# ============================================================
WEATHERS = ["fog", "night", "rain", "snow"]
SPLIT_MODE = "val"   # "val" or "all" depending on your folder logic

IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = 0

SAVE_TXT = True
SAVE_CONF = True
# ============================================================


def get_split_folder():
    # adjust this if later you want a different folder policy
    return "val"


def collect_images(weather):
    split_name = get_split_folder()
    weather_dir = ACDC_IMAGE_ROOT / weather / split_name
    if not weather_dir.exists():
        print(f"[WARNING] Missing folder: {weather_dir}")
        return []

    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(weather_dir.rglob(ext))
    return sorted(files)


def main():
    model = YOLO(str(MODEL_PT))

    run_dir = PROJECT_ROOT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = run_dir / f"predictions_{SPLIT_MODE}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = []

    for weather in WEATHERS:
        image_files = collect_images(weather)
        print(f"\n=== {weather.upper()} ===")
        print(f"Images: {len(image_files)}")

        weather_pred_dir = pred_dir / weather
        weather_pred_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_path in enumerate(image_files, start=1):
            results = model.predict(
                source=str(img_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False,
                save=True,
                save_txt=SAVE_TXT,
                save_conf=SAVE_CONF,
                project=str(pred_dir),
                name=weather,
                exist_ok=True,
            )

            num_masks = 0
            avg_conf = 0.0

            if len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    confs = r.boxes.conf.cpu().numpy()
                    num_masks = len(confs)
                    avg_conf = float(confs.mean()) if len(confs) > 0 else 0.0

            per_image_rows.append({
                "weather": weather,
                "image_name": img_path.name,
                "image_path": str(img_path),
                "num_masks": num_masks,
                "avg_conf": avg_conf,
                "has_detection": int(num_masks > 0),
                "predicted_image_dir": str(weather_pred_dir),
            })

            if idx % 20 == 0 or idx == len(image_files):
                print(f"Processed {idx}/{len(image_files)}")

    df = pd.DataFrame(per_image_rows)

    per_image_csv = run_dir / f"acdc_lane_per_image_results_{SPLIT_MODE}.csv"
    weather_csv = run_dir / f"acdc_lane_weather_summary_{SPLIT_MODE}.csv"
    weather_json = run_dir / f"acdc_lane_weather_summary_{SPLIT_MODE}.json"

    df.to_csv(per_image_csv, index=False)

    summary_df = (
        df.groupby("weather")
        .agg(
            images=("image_name", "count"),
            detection_rate=("has_detection", "mean"),
            no_detection_rate=("has_detection", lambda s: 1.0 - s.mean()),
            avg_lane_per_image=("num_masks", "mean"),
            avg_conf=("avg_conf", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(weather_csv, index=False)

    summary_payload = summary_df.to_dict(orient="records")
    with open(weather_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nDone.")
    print(f"Combined per-image CSV: {per_image_csv}")
    print(f"Weather summary CSV   : {weather_csv}")
    print(f"Weather summary JSON  : {weather_json}")
    print(f"Prediction images dir : {pred_dir}")


if __name__ == "__main__":
    main()
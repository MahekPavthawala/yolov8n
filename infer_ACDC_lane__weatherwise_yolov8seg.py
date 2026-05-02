# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:40:36 2026

Step L4: Run YOLOv8n-seg lane model on ACDC weather folders.

What it does:
- loads trained lane segmentation weights
- runs inference on ACDC weather folders
- saves overlays, masks, txt labels
- supports val/test/all modes
- supports full-run or random sampled images

Author: mahek
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import random
import json
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================
MODEL_PT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs\yolov8n_bdd100k_lane_seg2\weights\best.pt"
)

ACDC_IMAGE_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\ACDC\rgb_anon\rgb_anon"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
)

RUN_NAME = "yolov8n_bdd100k_lane_seg2"
# ============================================================


# ============================================================
# SETTINGS
# ============================================================
WEATHERS = ["fog", "night", "rain", "snow"]
SPLIT_MODE = "all"      # "val", "test", or "all"

USE_RANDOM_SAMPLE = False
SAMPLE_IMAGES_PER_WEATHER = 40
SEED = 42

IMGSZ = 640
CONF = 0.15
IOU = 0.45
DEVICE = 0

SAVE_OVERLAYS = True
SAVE_TXT = True
SAVE_FIGURE_SAMPLES = True
FIGURE_SAMPLES_PER_WEATHER = 12
MASK_ALPHA = 0.35
# ============================================================


def find_images_recursive(root_dir):
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_files.extend(root_dir.rglob(ext))
    return sorted(image_files)


def filter_split(images, split_mode):
    if split_mode == "all":
        return images
    needle = f"{os.sep}{split_mode}{os.sep}"
    return [p for p in images if needle in str(p)]


def collect_weather_images(weather):
    weather_dir = ACDC_IMAGE_ROOT / weather
    if not weather_dir.exists():
        raise FileNotFoundError(f"Missing weather folder: {weather_dir}")

    all_images = find_images_recursive(weather_dir)
    chosen = filter_split(all_images, SPLIT_MODE)

    if len(chosen) == 0:
        raise RuntimeError(f"No images found for {weather} / {SPLIT_MODE}")

    if USE_RANDOM_SAMPLE:
        random.seed(SEED)
        if len(chosen) > SAMPLE_IMAGES_PER_WEATHER:
            chosen = random.sample(chosen, SAMPLE_IMAGES_PER_WEATHER)

    return sorted(chosen)


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def overlay_masks_on_image(image_bgr, masks):
    """
    Overlay predicted masks in green.
    """
    overlay = image_bgr.copy()

    if masks is None or len(masks) == 0:
        return overlay

    for m in masks:
        mask = (m > 0.5).astype(np.uint8)
        green = np.zeros_like(image_bgr, dtype=np.uint8)
        green[:, :, 1] = 255

        idx = mask.astype(bool)
        overlay[idx] = cv2.addWeighted(image_bgr[idx], 1 - MASK_ALPHA, green[idx], MASK_ALPHA, 0)

    return overlay


def save_yolo_txt(txt_path, result, img_w, img_h):
    """
    Save segmentation polygons in YOLO-like text format from Ultralytics result.
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        txt_path.write_text("", encoding="utf-8")
        return

    lines = []
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # xy is in pixel coords per mask polygon
    polys = result.masks.xy

    for cls_id, poly in zip(classes, polys):
        if poly is None or len(poly) < 3:
            continue

        coords = []
        for x, y in poly:
            coords.append(max(0.0, min(1.0, x / img_w)))
            coords.append(max(0.0, min(1.0, y / img_h)))

        if len(coords) >= 6:
            line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in coords)
            lines.append(line)

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def run_weather(model, weather, image_list):
    out_dir = PROJECT_ROOT / f"{RUN_NAME}_{weather}_{SPLIT_MODE}_lane_infer_quiet"
    overlay_dir = out_dir / "overlays"
    txt_dir = out_dir / "labels"
    fig_dir = out_dir / "figure_samples"

    out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_OVERLAYS:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_TXT:
        txt_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_FIGURE_SAMPLES:
        fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"\n=== {weather.upper()} ===")
    print(f"Images: {len(image_list)}")

    for idx, img_path in enumerate(image_list, start=1):
        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            save=False,
            save_txt=False,
            verbose=False,
            retina_masks=True,
        )

        r = results[0]

        num_det = 0
        confs = []
        if r.boxes is not None and len(r.boxes) > 0:
            num_det = len(r.boxes)
            confs = [float(c) for c in r.boxes.conf.tolist()]

        img_bgr = cv2.imread(str(img_path))
        pred_image_path = None

        if img_bgr is not None and SAVE_OVERLAYS:
            masks_np = None
            if r.masks is not None and r.masks.data is not None:
                masks_np = r.masks.data.cpu().numpy()

            overlay = overlay_masks_on_image(img_bgr, masks_np)
            pred_image_path = overlay_dir / img_path.name
            cv2.imwrite(str(pred_image_path), overlay)

        if img_bgr is not None and SAVE_TXT:
            h, w = img_bgr.shape[:2]
            save_yolo_txt(txt_dir / f"{img_path.stem}.txt", r, w, h)

        rows.append({
            "weather": weather,
            "split": SPLIT_MODE,
            "image_name": img_path.name,
            "image_path": str(img_path),
            "predicted_image_path": str(pred_image_path) if pred_image_path else None,
            "num_lane_detections": num_det,
            "has_detection": int(num_det > 0),
            "avg_confidence": safe_mean(confs),
            "max_confidence": max(confs) if confs else 0.0,
        })

        if idx % 20 == 0 or idx == len(image_list):
            print(f"Processed {idx}/{len(image_list)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{weather}_per_image_results.csv", index=False)

    with open(out_dir / f"{weather}_per_image_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Save a curated subset of figure samples
    if SAVE_FIGURE_SAMPLES and len(df) > 0:
        # prioritize images with detections
        det_df = df[df["has_detection"] == 1].copy()
        sample_df = det_df.head(FIGURE_SAMPLES_PER_WEATHER)
        for _, row in sample_df.iterrows():
            src = Path(row["predicted_image_path"])
            if src.exists():
                dst = fig_dir / src.name
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())

    summary = {
        "weather": weather,
        "split": SPLIT_MODE,
        "num_images": int(len(df)),
        "images_with_detection": int(df["has_detection"].sum()),
        "images_without_detection": int((df["has_detection"] == 0).sum()),
        "no_detection_rate": float((df["has_detection"] == 0).mean()),
        "avg_lane_detections_per_image": float(df["num_lane_detections"].mean()),
        "max_lane_detections_in_an_image": int(df["num_lane_detections"].max()),
        "avg_confidence": float(df["avg_confidence"].mean()),
        "output_dir": str(out_dir),
    }

    return df, summary


def main():
    if not MODEL_PT.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PT}")

    model = YOLO(str(MODEL_PT))

    all_summaries = []
    all_rows = []

    for weather in WEATHERS:
        image_list = collect_weather_images(weather)
        df, summary = run_weather(model, weather, image_list)
        all_rows.append(df)
        all_summaries.append(summary)

    combined_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    run_dir = PROJECT_ROOT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = run_dir / f"acdc_lane_per_image_results_{SPLIT_MODE}.csv"
    summary_csv = run_dir / f"acdc_lane_weather_summary_{SPLIT_MODE}.csv"
    summary_json = run_dir / f"acdc_lane_weather_summary_{SPLIT_MODE}.json"

    combined_df.to_csv(combined_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nDone.")
    print(f"Combined per-image CSV: {combined_csv}")
    print(f"Weather summary CSV   : {summary_csv}")
    print(f"Weather summary JSON  : {summary_json}")


if __name__ == "__main__":
    main()
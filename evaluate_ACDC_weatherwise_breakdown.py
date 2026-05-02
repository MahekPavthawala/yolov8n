# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:36:09 2026

Weather-wise evaluation of trained YOLOv8n on ACDC:
fog, night, rain, snow

Author: mahek
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from ultralytics import YOLO
import json


# ============================================================
# PATHS
# ============================================================
MODEL_PT = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc"
    r"\yolov8n_bdd100k_8cls\weights\best.pt"
)

WEATHERWISE_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_yolo_weatherwise_8cls"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc"
)

RUN_NAME = "yolov8n_bdd100k_8cls"
# ============================================================


IMGSZ = 640
BATCH = 16
DEVICE = 0

WEATHERS = ["fog", "night", "rain", "snow"]


def extract_metric(metrics, name, default=None):
    try:
        value = getattr(metrics.box, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return default


def get_yaml(weather):
    return WEATHERWISE_ROOT / weather / f"acdc_{weather}.yaml"


def evaluate_weather(model, weather):
    yaml_path = get_yaml(weather)

    print(f"\n================ {weather.upper()} EVAL ================\n")

    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=0,
        plots=True,
        save_json=True,
        verbose=True,
        project=str(PROJECT_ROOT),
        name=f"{RUN_NAME}_{weather}_eval",
    )

    result = {
        "weather": weather,
        "map50": extract_metric(metrics, "map50"),
        "map75": extract_metric(metrics, "map75"),
        "map50_95": extract_metric(metrics, "map"),
        "precision": extract_metric(metrics, "mp"),
        "recall": extract_metric(metrics, "mr"),
    }

    return result


def save_summary(results):
    out_file = (
        PROJECT_ROOT
        / RUN_NAME
        / "weatherwise_cross_domain_summary.json"
    )

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary written to: {out_file}")


def print_table(results):
    print("\n================ WEATHER-WISE REPORT ================\n")
    print(f"{'Weather':<10} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['weather']:<10}"
            f"{r['map50']:<12.6f}"
            f"{r['map50_95']:<12.6f}"
            f"{r['precision']:<12.6f}"
            f"{r['recall']:<12.6f}"
        )


def main():
    if not MODEL_PT.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PT}")

    model = YOLO(str(MODEL_PT))

    results = []

    for weather in WEATHERS:
        result = evaluate_weather(model, weather)
        results.append(result)

    save_summary(results)
    print_table(results)

    print("\nDone.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:27:51 2026

@author: mahek
"""

# -*- coding: utf-8 -*-
"""
Validation + test-only script for completed ACDC fine-tuning run.

This script does NOT train.
It:
1) finds best.pt or last.pt from an existing fine-tune run
2) validates on ACDC val_gt
3) predicts on ACDC test_public
4) saves a summary JSON

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
RUN_DIR = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc\yolov8n_bdd_to_acdc_finetune2"
)

ACDC_VAL_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\val_gt\acdc_val_gt.yaml"
)

ACDC_TEST_DIR = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\test_public\images\test"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc"
)
# ============================================================


# ============================================================
# SETTINGS
# ============================================================
IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 0

CONF_PRED = 0.25
IOU_PRED = 0.45
# ============================================================


def get_weights_dir():
    return RUN_DIR / "weights"


def get_best_ckpt():
    return get_weights_dir() / "best.pt"


def get_last_ckpt():
    return get_weights_dir() / "last.pt"


def choose_checkpoint():
    best_ckpt = get_best_ckpt()
    last_ckpt = get_last_ckpt()

    if best_ckpt.exists():
        print("\nUsing best checkpoint:")
        print(best_ckpt)
        return best_ckpt

    if last_ckpt.exists():
        print("\nUsing last checkpoint:")
        print(last_ckpt)
        return last_ckpt

    raise FileNotFoundError(
        f"No best.pt or last.pt found in:\n{get_weights_dir()}"
    )


def ensure_paths():
    required = [RUN_DIR, ACDC_VAL_YAML, ACDC_TEST_DIR]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path:\n{p}")


def extract_metric(metrics, name, default=None):
    try:
        value = getattr(metrics.box, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return default


def validate_on_acdc_val(model_path):
    print("\n================ VALIDATION ON ACDC VAL_GT ================\n")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(ACDC_VAL_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=f"{RUN_DIR.name}_val_acdc_final",
        plots=True,
        save_json=True,
        verbose=True,
    )

    return metrics


def predict_on_acdc_test(model_path):
    print("\n================ PREDICTION ON ACDC TEST_PUBLIC ================\n")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(ACDC_TEST_DIR),
        imgsz=IMGSZ,
        conf=CONF_PRED,
        iou=IOU_PRED,
        device=DEVICE,
        project=str(PROJECT_ROOT),
        name=f"{RUN_DIR.name}_predict_acdc_test_final",
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        verbose=True,
    )

    return results


def save_summary(model_path, metrics):
    summary = {
        "run_dir": str(RUN_DIR),
        "evaluation_checkpoint": str(model_path),
        "acdc_val_yaml": str(ACDC_VAL_YAML),
        "acdc_test_dir": str(ACDC_TEST_DIR),
        "metrics_acdc_val": {
            "precision": extract_metric(metrics, "mp"),
            "recall": extract_metric(metrics, "mr"),
            "map50": extract_metric(metrics, "map50"),
            "map75": extract_metric(metrics, "map75"),
            "map50_95": extract_metric(metrics, "map"),
        },
    }

    out_file = RUN_DIR / "acdc_validation_and_test_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to:\n{out_file}")


def print_quick_report(model_path, metrics):
    print("\n================ QUICK REPORT ================\n")
    print(f"Checkpoint : {model_path}")
    print(f"Precision  : {extract_metric(metrics, 'mp')}")
    print(f"Recall     : {extract_metric(metrics, 'mr')}")
    print(f"mAP50      : {extract_metric(metrics, 'map50')}")
    print(f"mAP50-95   : {extract_metric(metrics, 'map')}")


def main():
    ensure_paths()

    model_path = choose_checkpoint()
    metrics = validate_on_acdc_val(model_path)
    predict_on_acdc_test(model_path)

    save_summary(model_path, metrics)
    print_quick_report(model_path, metrics)

    print("\nDone.")
    print(f"\nRun folder:\n{RUN_DIR}")


if __name__ == "__main__":
    main()
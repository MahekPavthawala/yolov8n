# -*- coding: utf-8 -*-
"""
Resume-aware fine-tuning of YOLOv8n from BDD100K -> ACDC train_gt,
then validation on ACDC val_gt, then prediction on ACDC test_public.

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
# Starting checkpoint = best BDD-trained object detector
BASE_MODEL = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc"
    r"\yolov8n_bdd100k_8cls\weights\best.pt"
)

ACDC_TRAIN_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\train_gt\acdc_train_gt.yaml"
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

RUN_NAME = "yolov8n_bdd_to_acdc_finetune_revamp"
# ============================================================


# ============================================================
# TRAIN SETTINGS
# ============================================================
EPOCHS = 20
IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 0
PATIENCE = 10
SAVE_PERIOD = 1
SEED = 42

CONF_PRED = 0.25
IOU_PRED = 0.45
# ============================================================


def get_run_dir():
    return PROJECT_ROOT / RUN_NAME


def get_weights_dir():
    return get_run_dir() / "weights"


def get_last_ckpt():
    return get_weights_dir() / "last.pt"


def get_best_ckpt():
    return get_weights_dir() / "best.pt"


def ensure_paths():
    required = [BASE_MODEL, ACDC_TRAIN_YAML, ACDC_VAL_YAML, ACDC_TEST_DIR]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path:\n{p}")


def write_training_config():
    cfg = {
        "base_model": str(BASE_MODEL),
        "acdc_train_yaml": str(ACDC_TRAIN_YAML),
        "acdc_val_yaml": str(ACDC_VAL_YAML),
        "acdc_test_dir": str(ACDC_TEST_DIR),
        "project_root": str(PROJECT_ROOT),
        "run_name": RUN_NAME,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "patience": PATIENCE,
        "save_period": SAVE_PERIOD,
        "seed": SEED,
        "conf_pred": CONF_PRED,
        "iou_pred": IOU_PRED,
    }

    out_path = get_run_dir() / "training_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Training config saved to:\n{out_path}")


def build_model():
    last_ckpt = get_last_ckpt()

    if last_ckpt.exists():
        print("\n================ RESUMING ACDC FINE-TUNING ================\n")
        print(f"Checkpoint:\n{last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        print("\n================ STARTING ACDC FINE-TUNING ================\n")
        print(f"Starting from BDD-trained checkpoint:\n{BASE_MODEL}")
        model = YOLO(str(BASE_MODEL))
        resume_flag = False

    return model, resume_flag


def train_or_resume():
    model, resume_flag = build_model()
    write_training_config()

    results = model.train(
        data=str(ACDC_TRAIN_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=RUN_NAME,
        patience=PATIENCE,
        save=True,
        save_period=SAVE_PERIOD,
        seed=SEED,
        plots=True,
        resume=resume_flag,
        verbose=True,
    )
    return results


def choose_eval_weights():
    best_ckpt = get_best_ckpt()
    last_ckpt = get_last_ckpt()

    if best_ckpt.exists():
        return best_ckpt
    if last_ckpt.exists():
        return last_ckpt

    raise FileNotFoundError("No best.pt or last.pt found after fine-tuning.")


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
        name=f"{RUN_NAME}_val_acdc",
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
        name=f"{RUN_NAME}_predict_acdc_test",
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        verbose=True,
    )

    return results


def extract_metric(metrics, name, default=None):
    try:
        value = getattr(metrics.box, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return default


def save_summary(model_path, metrics):
    summary = {
        "run_name": RUN_NAME,
        "starting_checkpoint": str(BASE_MODEL),
        "evaluation_checkpoint": str(model_path),
        "train_yaml": str(ACDC_TRAIN_YAML),
        "val_yaml": str(ACDC_VAL_YAML),
        "test_dir": str(ACDC_TEST_DIR),
        "metrics_acdc_val": {
            "precision": extract_metric(metrics, "mp"),
            "recall": extract_metric(metrics, "mr"),
            "map50": extract_metric(metrics, "map50"),
            "map75": extract_metric(metrics, "map75"),
            "map50_95": extract_metric(metrics, "map"),
        },
    }

    out_file = get_run_dir() / "acdc_finetune_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to:\n{out_file}")


def print_quick_report(model_path, metrics):
    print("\n================ QUICK REPORT ================\n")
    print(f"Evaluation checkpoint: {model_path}")
    print(f"Precision   : {extract_metric(metrics, 'mp')}")
    print(f"Recall      : {extract_metric(metrics, 'mr')}")
    print(f"mAP50       : {extract_metric(metrics, 'map50')}")
    print(f"mAP50-95    : {extract_metric(metrics, 'map')}")


def main():
    ensure_paths()

    train_or_resume()

    model_path = choose_eval_weights()
    metrics = validate_on_acdc_val(model_path)
    predict_on_acdc_test(model_path)

    save_summary(model_path, metrics)
    print_quick_report(model_path, metrics)

    print("\nDone.")
    print(f"\nMain run folder:\n{get_run_dir()}")


if __name__ == "__main__":
    main()
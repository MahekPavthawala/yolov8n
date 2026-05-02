# -*- coding: utf-8 -*-
"""
Single-folder, resume-aware YOLOv8n training on ACDC,
validation on ACDC val_gt, and external testing on DAWN.

Added features
--------------
- Explicit RESUME_CKPT support
- ACDC validation plots + confusion matrix
- DAWN evaluation plots + confusion matrix
- DAWN prediction images with bounding boxes, class names, and confidence
- Summary JSON + comparison CSV

Requirements
------------
- ACDC train_gt already converted to YOLO format
- ACDC val_gt already converted to YOLO format
- DAWN already converted to YOLO format with dawn_8cls.yaml

IMPORTANT FOR DAWN YAML
-----------------------
Your dawn_8cls.yaml should contain:

path: C:/UDMercy/Semester 1/Project/dawn_yolo_8cls
train: images/test
val: images/test
test: images/test

names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
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
BASE_MODEL = "yolov8n.pt"

ACDC_TRAIN_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\train_gt\acdc_train_gt.yaml"
)

ACDC_VAL_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\val_gt\acdc_val_gt.yaml"
)

DAWN_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\dawn_yolo_8cls\dawn_8cls.yaml"
)

DAWN_TEST_DIR = Path(
    r"C:\UDMercy\Semester 1\Project\dawn_yolo_8cls\images\test"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_acdc_only"
)

RUN_NAME = "yolov8n_acdc_train_acdcval_dawn_test"

# EXPLICIT RESUME CHECKPOINT
# If this exists, the script will resume from here first.
RESUME_CKPT = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_acdc_only"
    r"\yolov8n_acdc_train_acdcval_dawn_test\weights\last.pt"
)

# ============================================================
# SETTINGS
# ============================================================
EPOCHS = 30
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


def ensure_paths():
    required = [ACDC_TRAIN_YAML, ACDC_VAL_YAML, DAWN_YAML, DAWN_TEST_DIR]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path:\n{p}")


def get_run_dir() -> Path:
    return PROJECT_ROOT / RUN_NAME


def get_weights_dir() -> Path:
    return get_run_dir() / "weights"


def get_last_ckpt() -> Path:
    return get_weights_dir() / "last.pt"


def extract_metric(metrics, name, default=None):
    try:
        value = getattr(metrics.box, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return default


def write_training_config(run_dir: Path):
    cfg = {
        "base_model": BASE_MODEL,
        "resume_ckpt": str(RESUME_CKPT),
        "acdc_train_yaml": str(ACDC_TRAIN_YAML),
        "acdc_val_yaml": str(ACDC_VAL_YAML),
        "dawn_yaml": str(DAWN_YAML),
        "dawn_test_dir": str(DAWN_TEST_DIR),
        "project_root": str(PROJECT_ROOT),
        "run_name": RUN_NAME,
        "run_dir": str(run_dir),
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

    out_path = run_dir / "training_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Training config saved to:\n{out_path}")


def build_model():
    # 1) First priority = explicit checkpoint
    if RESUME_CKPT.exists():
        print("\n================ RESUMING FROM EXPLICIT CHECKPOINT ================\n")
        print(f"Checkpoint:\n{RESUME_CKPT}")
        model = YOLO(str(RESUME_CKPT))
        resume_flag = True
        return model, resume_flag

    # 2) Second priority = default run folder checkpoint
    last_ckpt = get_last_ckpt()
    if last_ckpt.exists():
        print("\n================ RESUMING ACDC TRAINING ================\n")
        print(f"Checkpoint:\n{last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
        return model, resume_flag

    # 3) Otherwise start fresh
    print("\n================ STARTING ACDC TRAINING ================\n")
    print(f"Starting from pretrained model:\n{BASE_MODEL}")
    model = YOLO(BASE_MODEL)
    resume_flag = False
    return model, resume_flag


def train_or_resume():
    model, resume_flag = build_model()

    results = model.train(
        data=str(ACDC_TRAIN_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=RUN_NAME,
        exist_ok=True,   # allow same folder when resuming
        patience=PATIENCE,
        save=True,
        save_period=SAVE_PERIOD,
        seed=SEED,
        plots=True,
        resume=resume_flag,
        verbose=True,
    )

    # Use actual save dir from Ultralytics
    run_dir = Path(model.trainer.save_dir)
    write_training_config(run_dir)
    return results, run_dir


def choose_eval_weights(run_dir: Path) -> Path:
    best_ckpt = run_dir / "weights" / "best.pt"
    last_ckpt = run_dir / "weights" / "last.pt"

    if best_ckpt.exists():
        return best_ckpt
    if last_ckpt.exists():
        return last_ckpt

    raise FileNotFoundError(f"No best.pt or last.pt found in:\n{run_dir / 'weights'}")

def find_existing_checkpoint():
    best_ckpt = get_run_dir() / "weights" / "best.pt"
    last_ckpt = get_run_dir() / "weights" / "last.pt"

    if best_ckpt.exists():
        print("\n================ FOUND EXISTING BEST CHECKPOINT ================\n")
        print(best_ckpt)
        return best_ckpt, get_run_dir()

    if last_ckpt.exists():
        print("\n================ FOUND EXISTING LAST CHECKPOINT ================\n")
        print(last_ckpt)
        return last_ckpt, get_run_dir()

    if RESUME_CKPT.exists():
        print("\n================ FOUND EXPLICIT RESUME CHECKPOINT ================\n")
        print(RESUME_CKPT)
        return RESUME_CKPT, RESUME_CKPT.parent.parent

    return None, None

def validate_on_acdc_val(model_path: Path, run_dir: Path):
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
        name=f"{run_dir.name}_val_acdc",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )
    return metrics


def evaluate_on_dawn(model_path: Path, run_dir: Path):
    print("\n================ EXTERNAL EVALUATION ON DAWN ================\n")

    model = YOLO(str(model_path))

    # IMPORTANT:
    # DAWN YAML should define train/val/test all as images/test.
    # Using split="val" is the safest workaround in Ultralytics.
    metrics = model.val(
        data=str(DAWN_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=f"{run_dir.name}_val_dawn",
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=True,
    )
    return metrics


def predict_on_dawn(model_path: Path, run_dir: Path):
    print("\n================ DAWN PREDICTION IMAGES ================\n")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(DAWN_TEST_DIR),
        imgsz=IMGSZ,
        conf=CONF_PRED,
        iou=IOU_PRED,
        device=DEVICE,
        project=str(PROJECT_ROOT),
        name=f"{run_dir.name}_predict_dawn",
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        verbose=True,
    )
    return results


def save_summary(model_path: Path, acdc_metrics, dawn_metrics, run_dir: Path):
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "starting_checkpoint": BASE_MODEL,
        "resume_checkpoint": str(RESUME_CKPT),
        "evaluation_checkpoint": str(model_path),
        "train_yaml": str(ACDC_TRAIN_YAML),
        "acdc_val_yaml": str(ACDC_VAL_YAML),
        "dawn_yaml": str(DAWN_YAML),
        "dawn_test_dir": str(DAWN_TEST_DIR),
        "metrics_acdc_val": {
            "precision": extract_metric(acdc_metrics, "mp"),
            "recall": extract_metric(acdc_metrics, "mr"),
            "map50": extract_metric(acdc_metrics, "map50"),
            "map75": extract_metric(acdc_metrics, "map75"),
            "map50_95": extract_metric(acdc_metrics, "map"),
        },
        "metrics_dawn_test": {
            "precision": extract_metric(dawn_metrics, "mp"),
            "recall": extract_metric(dawn_metrics, "mr"),
            "map50": extract_metric(dawn_metrics, "map50"),
            "map75": extract_metric(dawn_metrics, "map75"),
            "map50_95": extract_metric(dawn_metrics, "map"),
        },
        "output_dirs": {
            "training_run_dir": str(run_dir),
            "acdc_val_eval_dir": str(PROJECT_ROOT / f"{run_dir.name}_val_acdc"),
            "dawn_eval_dir": str(PROJECT_ROOT / f"{run_dir.name}_val_dawn"),
            "dawn_prediction_dir": str(PROJECT_ROOT / f"{run_dir.name}_predict_dawn"),
        }
    }

    out_file = run_dir / "acdc_to_dawn_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to:\n{out_file}")
    return summary


def make_comparison_csv(run_dir: Path, summary: dict):
    rows = [
        {
            "evaluation_dataset": "ACDC_val_gt",
            "precision": summary["metrics_acdc_val"]["precision"],
            "recall": summary["metrics_acdc_val"]["recall"],
            "map50": summary["metrics_acdc_val"]["map50"],
            "map75": summary["metrics_acdc_val"]["map75"],
            "map50_95": summary["metrics_acdc_val"]["map50_95"],
        },
        {
            "evaluation_dataset": "DAWN_test",
            "precision": summary["metrics_dawn_test"]["precision"],
            "recall": summary["metrics_dawn_test"]["recall"],
            "map50": summary["metrics_dawn_test"]["map50"],
            "map75": summary["metrics_dawn_test"]["map75"],
            "map50_95": summary["metrics_dawn_test"]["map50_95"],
        },
    ]

    df = pd.DataFrame(rows)

    if len(df) == 2:
        for metric in ["precision", "recall", "map50", "map75", "map50_95"]:
            df[f"{metric}_drop_vs_acdc_val"] = None
            if pd.notna(df.loc[0, metric]) and pd.notna(df.loc[1, metric]):
                df.loc[1, f"{metric}_drop_vs_acdc_val"] = df.loc[1, metric] - df.loc[0, metric]

    out_csv = run_dir / "acdc_val_vs_dawn_test_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nComparison CSV written to:\n{out_csv}")
    return out_csv


def print_quick_report(model_path: Path, acdc_metrics, dawn_metrics, run_dir: Path):
    print("\n================ QUICK REPORT ================\n")
    print(f"Run folder        : {run_dir}")
    print(f"Checkpoint        : {model_path}")

    print("\nACDC val:")
    print(f"  Precision       : {extract_metric(acdc_metrics, 'mp')}")
    print(f"  Recall          : {extract_metric(acdc_metrics, 'mr')}")
    print(f"  mAP50           : {extract_metric(acdc_metrics, 'map50')}")
    print(f"  mAP50-95        : {extract_metric(acdc_metrics, 'map')}")

    print("\nDAWN test:")
    print(f"  Precision       : {extract_metric(dawn_metrics, 'mp')}")
    print(f"  Recall          : {extract_metric(dawn_metrics, 'mr')}")
    print(f"  mAP50           : {extract_metric(dawn_metrics, 'map50')}")
    print(f"  mAP50-95        : {extract_metric(dawn_metrics, 'map')}")

    print("\nGenerated output folders:")
    print(f"  Training        : {run_dir}")
    print(f"  ACDC val plots  : {PROJECT_ROOT / f'{run_dir.name}_val_acdc'}")
    print(f"  DAWN plots      : {PROJECT_ROOT / f'{run_dir.name}_val_dawn'}")
    print(f"  DAWN predictions: {PROJECT_ROOT / f'{run_dir.name}_predict_dawn'}")


def main():
    ensure_paths()
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    # If a finished checkpoint already exists, skip training completely
    model_path, run_dir = find_existing_checkpoint()

    if model_path is None:
        print("\nNo existing checkpoint found. Starting training...\n")
        _, run_dir = train_or_resume()
        model_path = choose_eval_weights(run_dir)
    else:
        print("\nSkipping training and using existing checkpoint.\n")
        write_training_config(run_dir)

    acdc_metrics = validate_on_acdc_val(model_path, run_dir)
    dawn_metrics = evaluate_on_dawn(model_path, run_dir)
    predict_on_dawn(model_path, run_dir)

    summary = save_summary(model_path, acdc_metrics, dawn_metrics, run_dir)
    make_comparison_csv(run_dir, summary)
    print_quick_report(model_path, acdc_metrics, dawn_metrics, run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
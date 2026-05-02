# -*- coding: utf-8 -*-
"""
Single-folder, resume-aware fine-tuning of YOLOv8n-seg
from BDD100K lane model -> ACDC pseudo-label dataset.

Pipeline
--------
1) Start from BDD-trained lane segmentation checkpoint
2) Fine-tune on ACDC pseudo-labels
3) Validate on pseudo-label dataset itself
4) Save weights, plots, results, summaries in one folder

Author: mahek
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================
BASE_MODEL = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
    r"\yolov8n_bdd100k_lane_seg2\weights\best.pt"
)

PSEUDO_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\acdc_lane_pseudo\acdc_lane_pseudo.yaml"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
)

RUN_NAME = "yolov8n_bdd_lane_to_acdc_pseudo"

RESUME_CKPT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
    r"\yolov8n_bdd_lane_to_acdc_pseudo\weights\last.pt"
)
# ============================================================


# ============================================================
# SETTINGS
# ============================================================
EPOCHS = 20
IMGSZ = 640
BATCH = 8
DEVICE = 0
WORKERS = 0
PATIENCE = 8
SAVE_PERIOD = 1
SEED = 42
# ============================================================


def ensure_paths():
    required = [BASE_MODEL, PSEUDO_YAML]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path:\n{p}")


def get_run_dir() -> Path:
    return PROJECT_ROOT / RUN_NAME


def get_weights_dir() -> Path:
    return get_run_dir() / "weights"


def get_last_ckpt() -> Path:
    return get_weights_dir() / "last.pt"


def extract_metric(metrics, section, name, default=None):
    try:
        obj = getattr(metrics, section)
        value = getattr(obj, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return default


def write_training_config(run_dir: Path):
    cfg = {
        "base_model": str(BASE_MODEL),
        "resume_ckpt": str(RESUME_CKPT),
        "pseudo_yaml": str(PSEUDO_YAML),
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
    }

    out_path = run_dir / "training_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Training config saved to:\n{out_path}")


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


def build_model():
    if RESUME_CKPT.exists():
        print("\n================ RESUMING FROM EXPLICIT CHECKPOINT ================\n")
        print(f"Checkpoint:\n{RESUME_CKPT}")
        model = YOLO(str(RESUME_CKPT))
        resume_flag = True
        return model, resume_flag

    last_ckpt = get_last_ckpt()
    if last_ckpt.exists():
        print("\n================ RESUMING PSEUDO-LABEL FINE-TUNING ================\n")
        print(f"Checkpoint:\n{last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
        return model, resume_flag

    print("\n================ STARTING PSEUDO-LABEL FINE-TUNING ================\n")
    print(f"Starting from BDD-trained lane model:\n{BASE_MODEL}")
    model = YOLO(str(BASE_MODEL))
    resume_flag = False
    return model, resume_flag


def train_or_resume():
    model, resume_flag = build_model()

    results = model.train(
        data=str(PSEUDO_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        patience=PATIENCE,
        save=True,
        save_period=SAVE_PERIOD,
        seed=SEED,
        plots=True,
        resume=resume_flag,
        verbose=True,
    )

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


def validate_on_pseudo_val(model_path: Path, run_dir: Path):
    print("\n================ VALIDATION ON ACDC PSEUDO DATASET ================\n")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(PSEUDO_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=f"{run_dir.name}_val_pseudo",
        exist_ok=True,
        plots=True,
        save_json=False,
        verbose=True,
    )
    return metrics


def save_summary(model_path: Path, metrics, run_dir: Path):
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "starting_checkpoint": str(BASE_MODEL),
        "resume_checkpoint": str(RESUME_CKPT),
        "evaluation_checkpoint": str(model_path),
        "pseudo_yaml": str(PSEUDO_YAML),
        "metrics": {
            "box_precision": extract_metric(metrics, "box", "mp"),
            "box_recall": extract_metric(metrics, "box", "mr"),
            "box_map50": extract_metric(metrics, "box", "map50"),
            "box_map50_95": extract_metric(metrics, "box", "map"),
            "mask_precision": extract_metric(metrics, "seg", "mp"),
            "mask_recall": extract_metric(metrics, "seg", "mr"),
            "mask_map50": extract_metric(metrics, "seg", "map50"),
            "mask_map50_95": extract_metric(metrics, "seg", "map"),
        },
    }

    out_file = run_dir / "acdc_pseudo_finetune_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to:\n{out_file}")
    return summary


def print_quick_report(model_path: Path, metrics, run_dir: Path):
    print("\n================ QUICK REPORT ================\n")
    print(f"Run folder        : {run_dir}")
    print(f"Checkpoint        : {model_path}")
    print(f"Mask Precision    : {extract_metric(metrics, 'seg', 'mp')}")
    print(f"Mask Recall       : {extract_metric(metrics, 'seg', 'mr')}")
    print(f"Mask mAP50        : {extract_metric(metrics, 'seg', 'map50')}")
    print(f"Mask mAP50-95     : {extract_metric(metrics, 'seg', 'map')}")


def main():
    ensure_paths()
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    model_path, run_dir = find_existing_checkpoint()

    if model_path is None:
        print("\nNo existing checkpoint found. Starting fine-tuning...\n")
        _, run_dir = train_or_resume()
        model_path = choose_eval_weights(run_dir)
    else:
        print("\nSkipping training and using existing checkpoint.\n")
        write_training_config(run_dir)

    metrics = validate_on_pseudo_val(model_path, run_dir)
    save_summary(model_path, metrics, run_dir)
    print_quick_report(model_path, metrics, run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
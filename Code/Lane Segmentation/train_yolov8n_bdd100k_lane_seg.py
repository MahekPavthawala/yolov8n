# -*- coding: utf-8 -*-
"""
Resume-aware YOLOv8n-seg training on BDD100K lane segmentation.

Author: mahek
"""

from pathlib import Path
from ultralytics import YOLO
import json


# ============================================================
# PATHS
# ============================================================
DATA_YAML = Path(
    r"C:\UDMercy\Semester 1\Project\bdd100k_lane_yolo_seg\bdd100k_lane_seg.yaml"
)

PROJECT_ROOT = Path(
    r"C:\UDMercy\Semester 1\Project\lane_runs"
)

RUN_NAME = "yolov8n_bdd100k_lane_seg"

BASE_MODEL = "yolov8n-seg.pt"
# ============================================================


# ============================================================
# TRAIN SETTINGS
# ============================================================
EPOCHS = 30
IMGSZ = 640
BATCH = 8
DEVICE = 0
WORKERS = 4

PATIENCE = 20
SAVE_PERIOD = 1
# ============================================================


def get_run_dir():
    return PROJECT_ROOT / RUN_NAME


def get_last_ckpt():
    return get_run_dir() / "weights" / "last.pt"


def get_best_ckpt():
    return get_run_dir() / "weights" / "best.pt"


def write_training_config():
    cfg = {
        "data_yaml": str(DATA_YAML),
        "project_root": str(PROJECT_ROOT),
        "run_name": RUN_NAME,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "base_model": BASE_MODEL,
    }

    out_path = get_run_dir() / "training_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Training config saved: {out_path}")


def build_model():
    last_ckpt = get_last_ckpt()

    if last_ckpt.exists():
        print("\n================ RESUMING TRAINING ================\n")
        print(f"Resuming from:\n{last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        print("\n================ NEW TRAINING ================\n")
        print(f"Starting from pretrained model:\n{BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        resume_flag = False

    return model, resume_flag


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset yaml:\n{DATA_YAML}")

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    model, resume_flag = build_model()

    write_training_config()

    results = model.train(
        data=str(DATA_YAML),
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
        plots=True,
        resume=resume_flag,
        verbose=True,
    )

    print("\n================ TRAINING COMPLETE ================\n")
    print(f"Run folder:\n{get_run_dir()}")

    best_ckpt = get_best_ckpt()
    last_ckpt = get_last_ckpt()

    if best_ckpt.exists():
        print(f"Best weights:\n{best_ckpt}")

    if last_ckpt.exists():
        print(f"Last weights:\n{last_ckpt}")

    return results


if __name__ == "__main__":
    main()
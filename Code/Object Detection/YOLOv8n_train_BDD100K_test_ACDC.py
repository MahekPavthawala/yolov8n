# -*- coding: utf-8 -*-
"""
Resume-aware YOLO training/evaluation pipeline

What this script does:
1) Train on BDD100K 8-class YOLO dataset
2) If interrupted, resume from last.pt automatically
3) If training already finished, skip retraining
4) Validate on BDD100K val
5) Validate on ACDC val_gt
6) Predict on ACDC public test
7) Save a summary JSON

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
BDD_YAML = Path(r"C:\UDMercy\Semester 1\Project\bdd100k_yolo_8cls\bdd100k.yaml")
ACDC_VAL_YAML = Path(r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\val_gt\acdc_val_gt.yaml")
ACDC_TEST_DIR = Path(r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\test_public\images\test")

PROJECT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc")
RUN_NAME = "yolov8n_bdd100k_8cls"


BASE_WEIGHTS = "yolov8n.pt"
# ============================================================


# ============================================================
# TRAINING SETTINGS
# ============================================================
EPOCHS = 30
IMGSZ = 640
BATCH = 16
DEVICE = 0          # use 0 for GPU, or "cpu"
WORKERS = 8
PATIENCE = 20
SEED = 42

SAVE_PERIOD = 5
CONF_PRED = 0.25
IOU_PRED = 0.45
# ============================================================


def ensure_paths():
    required = [BDD_YAML, ACDC_VAL_YAML, ACDC_TEST_DIR]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")


def get_run_dir():
    return PROJECT_ROOT / RUN_NAME


def get_weights_dir():
    return get_run_dir() / "weights"


def get_last_weights():
    return get_weights_dir() / "last.pt"


def get_best_weights():
    return get_weights_dir() / "best.pt"


def get_args_yaml():
    return get_run_dir() / "args.yaml"


def is_training_already_completed():
    
    run_dir = get_run_dir()
    best_pt = get_best_weights()
    last_pt = get_last_weights()
    results_csv = run_dir / "results.csv"

    return best_pt.exists() and last_pt.exists() and results_csv.exists()


def start_fresh_training():
    """
    Start a brand-new training run from BASE_WEIGHTS.
    """
    print("\n================ STARTING FRESH TRAINING ON BDD100K ================\n")

    model = YOLO(BASE_WEIGHTS)

    model.train(
        data=str(BDD_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        seed=SEED,
        project=str(PROJECT_ROOT),
        name=RUN_NAME,
        save=True,
        save_period=SAVE_PERIOD,
        pretrained=True,
        verbose=True,
        plots=True,
    )


def resume_training():
    """
    Resume interrupted training from last.pt.

    This preserves:
    - model weights
    - biases
    - optimizer state
    - scheduler state
    - epoch progress
    """
    last_pt = get_last_weights()

    if not last_pt.exists():
        raise FileNotFoundError(f"Cannot resume because checkpoint does not exist: {last_pt}")

    print("\n================ RESUMING TRAINING FROM last.pt ================\n")
    print(f"Checkpoint: {last_pt}\n")

    model = YOLO(str(last_pt))
    model.train(resume=True)


def train_or_resume_if_needed():
    """
    Logic:
    1) If training is already finished -> skip retraining
    2) Else if last.pt exists -> resume
    3) Else -> start fresh
    """
    if is_training_already_completed():
        print("\n================ TRAINING STATUS ================\n")
        print("Training appears to be already completed.")
        print("Skipping retraining and continuing to validation/prediction.\n")
        return

    if get_last_weights().exists():
        resume_training()
    else:
        start_fresh_training()


def choose_eval_weights():
    """
    For evaluation, prefer best.pt. Fall back to last.pt.
    """
    best_pt = get_best_weights()
    last_pt = get_last_weights()

    if best_pt.exists():
        return best_pt
    if last_pt.exists():
        return last_pt

    raise FileNotFoundError("Could not find best.pt or last.pt after training/resume.")


def validate_on_bdd(model_path):
    print("\n================ VALIDATION ON BDD100K ================\n")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(BDD_YAML),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        split="val",
        project=str(PROJECT_ROOT),
        name=f"{RUN_NAME}_val_bdd",
        plots=True,
        save_json=True,
        verbose=True,
    )

    return metrics


def validate_on_acdc_val(model_path):
    print("\n================ VALIDATION ON ACDC VAL_GT ================\n")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(ACDC_VAL_YAML),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        split="val",
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
        verbose=True,
    )

    return results


def extract_metric(metrics, name, default=None):
    """
    Safe getter for Ultralytics metrics fields.
    """
    try:
        value = getattr(metrics.box, name)
        if callable(value):
            value = value()
        return value
    except Exception:
        return default


def save_summary(eval_model_path, bdd_metrics, acdc_metrics):
    summary = {
        "run_name": RUN_NAME,
        "base_weights_for_fresh_start": BASE_WEIGHTS,
        "evaluation_model_path": str(eval_model_path),
        "train_data": str(BDD_YAML),
        "acdc_val_data": str(ACDC_VAL_YAML),
        "acdc_test_source": str(ACDC_TEST_DIR),
        "run_dir": str(get_run_dir()),
        "weights_dir": str(get_weights_dir()),
        "checkpoints": {
            "best_pt_exists": get_best_weights().exists(),
            "last_pt_exists": get_last_weights().exists(),
            "best_pt": str(get_best_weights()),
            "last_pt": str(get_last_weights()),
        },
        "settings": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "workers": WORKERS,
            "patience": PATIENCE,
            "seed": SEED,
            "save_period": SAVE_PERIOD,
            "conf_pred": CONF_PRED,
            "iou_pred": IOU_PRED,
        },
        "bdd_val": {
            "map50": extract_metric(bdd_metrics, "map50"),
            "map75": extract_metric(bdd_metrics, "map75"),
            "map50_95": extract_metric(bdd_metrics, "map"),
            "mp": extract_metric(bdd_metrics, "mp"),
            "mr": extract_metric(bdd_metrics, "mr"),
        },
        "acdc_val": {
            "map50": extract_metric(acdc_metrics, "map50"),
            "map75": extract_metric(acdc_metrics, "map75"),
            "map50_95": extract_metric(acdc_metrics, "map"),
            "mp": extract_metric(acdc_metrics, "mp"),
            "mr": extract_metric(acdc_metrics, "mr"),
        },
    }

    out_file = get_run_dir() / "cross_domain_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {out_file}")


def print_quick_report(eval_model_path, bdd_metrics, acdc_metrics):
    bdd_map50 = extract_metric(bdd_metrics, "map50")
    bdd_map = extract_metric(bdd_metrics, "map")
    acdc_map50 = extract_metric(acdc_metrics, "map50")
    acdc_map = extract_metric(acdc_metrics, "map")

    print("\n================ QUICK REPORT ================\n")
    print(f"Evaluation weights : {eval_model_path}")
    print(f"BDD100K val        : mAP50={bdd_map50}, mAP50-95={bdd_map}")
    print(f"ACDC val_gt        : mAP50={acdc_map50}, mAP50-95={acdc_map}")

    if bdd_map is not None and acdc_map is not None:
        try:
            drop = bdd_map - acdc_map
            print(f"Cross-domain drop (mAP50-95): {drop:.6f}")
        except Exception:
            pass


def main():
    ensure_paths()

    # Train fresh / resume / skip
    train_or_resume_if_needed()

    # Use best weights for evaluation if available
    eval_model_path = choose_eval_weights()

    # Continue remaining pipeline
    bdd_metrics = validate_on_bdd(eval_model_path)
    acdc_metrics = validate_on_acdc_val(eval_model_path)
    predict_on_acdc_test(eval_model_path)

    save_summary(eval_model_path, bdd_metrics, acdc_metrics)
    print_quick_report(eval_model_path, bdd_metrics, acdc_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
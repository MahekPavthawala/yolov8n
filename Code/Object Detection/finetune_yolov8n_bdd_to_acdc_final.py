# -*- coding: utf-8 -*-
"""
Fine-tuning of YOLOv8n from BDD100K -> ACDC train_gt,
then validation on ACDC val_gt, prediction on ACDC test_public, and CSV comparison
against the BDD100K->ACDC baseline.

What this script does
---------------------
1) Starts from the BDD-trained checkpoint (best.pt)
2) Fine-tunes on ACDC train_gt
3) Validates on ACDC val_gt
4) Predicts on ACDC test_public and saves images with boxes/classes/confidence
5) Writes all config, summaries, plots, weights, and comparison CSVs into ONE folder
6) Is resume-aware:
   - if last.pt exists in the run folder, it resumes from it
   - otherwise it starts from the BDD-trained checkpoint

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
BASE_MODEL = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc"
    r"\yolov8n_bdd100k_8cls\weights\best.pt"
)

BASELINE_SUMMARY_JSON = Path(
    r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc"
    r"\yolov8n_bdd100k_8cls\cross_domain_summary.json"
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

RUN_NAME = "yolov8n_bdd_to_acdc_adapted_final"
# ============================================================


# ============================================================
# TRAIN SETTINGS
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
    required = [BASE_MODEL, ACDC_TRAIN_YAML, ACDC_VAL_YAML, ACDC_TEST_DIR]
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


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_training_config(run_dir: Path):
    cfg = {
        "base_model": str(BASE_MODEL),
        "baseline_summary_json": str(BASELINE_SUMMARY_JSON),
        "acdc_train_yaml": str(ACDC_TRAIN_YAML),
        "acdc_val_yaml": str(ACDC_VAL_YAML),
        "acdc_test_dir": str(ACDC_TEST_DIR),
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

    results = model.train(
        data=str(ACDC_TRAIN_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_ROOT),
        name=RUN_NAME,
        exist_ok=False,
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


def predict_on_acdc_test(model_path: Path, run_dir: Path):
    print("\n================ PREDICTION ON ACDC TEST_PUBLIC ================\n")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(ACDC_TEST_DIR),
        imgsz=IMGSZ,
        conf=CONF_PRED,
        iou=IOU_PRED,
        device=DEVICE,
        project=str(PROJECT_ROOT),
        name=f"{run_dir.name}_predict_acdc_test",
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        verbose=True,
    )
    return results


def save_summary(model_path: Path, metrics, run_dir: Path):
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
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

    out_file = run_dir / "acdc_finetune_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to:\n{out_file}")
    return summary, out_file


def make_comparison_csv(run_dir: Path, adapted_summary: dict):
    baseline = load_json(BASELINE_SUMMARY_JSON)
    rows = []

    if baseline is not None:
        b = baseline.get("acdc_val", {})
        rows.append({
            "model_variant": "baseline_bdd_to_acdc_val",
            "training_domain": "BDD100K",
            "evaluation_domain": "ACDC_val_gt",
            "precision": b.get("mp"),
            "recall": b.get("mr"),
            "map50": b.get("map50"),
            "map75": b.get("map75"),
            "map50_95": b.get("map50_95"),
        })

    a = adapted_summary["metrics_acdc_val"]
    rows.append({
        "model_variant": "adapted_bdd_to_acdc_train_then_val",
        "training_domain": "BDD100K + ACDC_train_gt",
        "evaluation_domain": "ACDC_val_gt",
        "precision": a["precision"],
        "recall": a["recall"],
        "map50": a["map50"],
        "map75": a["map75"],
        "map50_95": a["map50_95"],
    })

    df = pd.DataFrame(rows)

    if len(df) >= 2:
        base = df.iloc[0]
        for metric in ["precision", "recall", "map50", "map75", "map50_95"]:
            df[f"{metric}_abs_gain_vs_baseline"] = None
            if pd.notna(base[metric]) and pd.notna(df.loc[1, metric]):
                df.loc[1, f"{metric}_abs_gain_vs_baseline"] = df.loc[1, metric] - base[metric]

    out_csv = run_dir / "baseline_vs_adapted_acdc_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nComparison CSV written to:\n{out_csv}")
    return out_csv


def print_quick_report(model_path: Path, metrics, run_dir: Path):
    print("\n================ QUICK REPORT ================\n")
    print(f"Run folder   : {run_dir}")
    print(f"Checkpoint   : {model_path}")
    print(f"Precision    : {extract_metric(metrics, 'mp')}")
    print(f"Recall       : {extract_metric(metrics, 'mr')}")
    print(f"mAP50        : {extract_metric(metrics, 'map50')}")
    print(f"mAP50-95     : {extract_metric(metrics, 'map')}")
    print(f"Test outputs : {PROJECT_ROOT / f'{run_dir.name}_predict_acdc_test'}")


def main():
    ensure_paths()

    results, run_dir = train_or_resume()

    if run_dir.name != RUN_NAME:
        raise RuntimeError(
            f"Ultralytics created a different folder name ({run_dir.name}). "
            f"Use a fresh RUN_NAME and rerun so everything stays in one folder."
        )

    model_path = choose_eval_weights(run_dir)
    metrics = validate_on_acdc_val(model_path, run_dir)
    predict_on_acdc_test(model_path, run_dir)

    adapted_summary, _ = save_summary(model_path, metrics, run_dir)
    make_comparison_csv(run_dir, adapted_summary)
    print_quick_report(model_path, metrics, run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

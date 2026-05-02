# -*- coding: utf-8 -*-
"""
End-to-end evaluation summarizer for the BDD100K -> ACDC YOLO project.

What it does
------------
1) Reads training/evaluation artifacts already produced by Ultralytics:
   - run_dir/results.csv
   - cross_domain_summary.json
   - weatherwise_cross_domain_summary.json
2) Optionally reads predictions.json files from:
   - *_val_bdd
   - *_fog_eval / *_night_eval / *_rain_eval / *_snow_eval
3) Creates:
   - a markdown report
   - CSV summary tables
   - bar charts for overall and weather-wise metrics
   - line chart from training results.csv if available

This script does NOT retrain the model. It only summarizes completed runs.

Author: mahek
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER PATHS - UPDATE IF NEEDED
# ============================================================
RUN_DIR = Path(r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc\yolov8n_bdd100k_8cls")
PROJECT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\yolo_runs_bdd_to_acdc")

# These are the sibling eval folders produced earlier
BDD_VAL_DIR = PROJECT_ROOT / "yolov8n_bdd100k_8cls_val_bdd"
FOG_EVAL_DIR = PROJECT_ROOT / "yolov8n_bdd100k_8cls_fog_eval"
NIGHT_EVAL_DIR = PROJECT_ROOT / "yolov8n_bdd100k_8cls_night_eval"
RAIN_EVAL_DIR = PROJECT_ROOT / "yolov8n_bdd100k_8cls_rain_eval"
SNOW_EVAL_DIR = PROJECT_ROOT / "yolov8n_bdd100k_8cls_snow_eval"

# Output folder for generated report assets
OUTPUT_DIR = RUN_DIR / "final_evaluation_package_yolov8n"
# ============================================================


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def pct_drop(src, dst):
    if src is None or dst is None or src == 0:
        return None
    return 100.0 * (src - dst) / src


def infer_weather_rank(weather_df, metric_col):
    df = weather_df.sort_values(metric_col, ascending=False)
    return df["weather"].tolist()


def create_bar_chart(df, xcol, ycol, title, out_path, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(df[xcol], df[ycol])
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def create_training_curve(results_csv, out_path):
    if not results_csv.exists():
        return False

    df = pd.read_csv(results_csv)
    cols = {c.strip(): c for c in df.columns}
    epoch_col = cols.get("epoch", None)

    map_col = None
    for candidate in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(B) "]:
        if candidate in df.columns:
            map_col = candidate
            break

    map50_col = None
    for candidate in ["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP50(B) "]:
        if candidate in df.columns:
            map50_col = candidate
            break

    if epoch_col is None or (map_col is None and map50_col is None):
        return False

    plt.figure(figsize=(8, 5))
    if map_col is not None:
        plt.plot(df[epoch_col], df[map_col], label="mAP50-95")
    if map50_col is not None:
        plt.plot(df[epoch_col], df[map50_col], label="mAP50")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("BDD100K validation metrics during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def summarize():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cross_summary = load_json(RUN_DIR / "cross_domain_summary.json")
    weather_summary = load_json(RUN_DIR / "weatherwise_cross_domain_summary.json")

    if cross_summary is None:
        raise FileNotFoundError(f"Missing cross_domain_summary.json in {RUN_DIR}")
    if weather_summary is None:
        raise FileNotFoundError(f"Missing weatherwise_cross_domain_summary.json in {RUN_DIR}")

    bdd = cross_summary.get("bdd_val", {})
    acdc = cross_summary.get("acdc_val", {})

    overall_df = pd.DataFrame([
        {
            "dataset": "BDD100K_val",
            "mAP50": safe_float(bdd.get("map50")),
            "mAP50_95": safe_float(bdd.get("map50_95")),
            "precision": safe_float(bdd.get("mp")),
            "recall": safe_float(bdd.get("mr")),
        },
        {
            "dataset": "ACDC_val_gt",
            "mAP50": safe_float(acdc.get("map50")),
            "mAP50_95": safe_float(acdc.get("map50_95")),
            "precision": safe_float(acdc.get("mp")),
            "recall": safe_float(acdc.get("mr")),
        },
    ])
    overall_df["mAP50_abs_drop_vs_bdd"] = overall_df["mAP50"].iloc[0] - overall_df["mAP50"]
    overall_df["mAP50_95_abs_drop_vs_bdd"] = overall_df["mAP50_95"].iloc[0] - overall_df["mAP50_95"]
    overall_df["mAP50_rel_drop_pct_vs_bdd"] = overall_df["mAP50"].apply(
        lambda x: pct_drop(overall_df["mAP50"].iloc[0], x)
    )
    overall_df["mAP50_95_rel_drop_pct_vs_bdd"] = overall_df["mAP50_95"].apply(
        lambda x: pct_drop(overall_df["mAP50_95"].iloc[0], x)
    )
    overall_df.to_csv(OUTPUT_DIR / "overall_summary_table.csv", index=False)

    weather_results = weather_summary if isinstance(weather_summary, list) else weather_summary.get("weather_results", [])
    weather_df = pd.DataFrame(weather_results)
    if weather_df.empty:
        raise ValueError("No weather_results found in weatherwise_cross_domain_summary.json")

    weather_df.rename(columns={"map50_95": "mAP50_95", "map50": "mAP50"}, inplace=True)
    weather_df["mAP50_abs_drop_vs_bdd"] = weather_df["mAP50"].apply(lambda x: overall_df["mAP50"].iloc[0] - x)
    weather_df["mAP50_95_abs_drop_vs_bdd"] = weather_df["mAP50_95"].apply(lambda x: overall_df["mAP50_95"].iloc[0] - x)
    weather_df["mAP50_rel_drop_pct_vs_bdd"] = weather_df["mAP50"].apply(lambda x: pct_drop(overall_df["mAP50"].iloc[0], x))
    weather_df["mAP50_95_rel_drop_pct_vs_bdd"] = weather_df["mAP50_95"].apply(lambda x: pct_drop(overall_df["mAP50_95"].iloc[0], x))
    weather_df.to_csv(OUTPUT_DIR / "weatherwise_summary_table.csv", index=False)

    create_bar_chart(overall_df, "dataset", "mAP50", "Overall domain comparison: mAP50", OUTPUT_DIR / "overall_map50_bar.png", "mAP50")
    create_bar_chart(overall_df, "dataset", "mAP50_95", "Overall domain comparison: mAP50-95", OUTPUT_DIR / "overall_map50_95_bar.png", "mAP50-95")
    create_bar_chart(weather_df.sort_values("mAP50", ascending=False), "weather", "mAP50", "ACDC weather-wise comparison: mAP50", OUTPUT_DIR / "weather_map50_bar.png", "mAP50")
    create_bar_chart(weather_df.sort_values("mAP50_95", ascending=False), "weather", "mAP50_95", "ACDC weather-wise comparison: mAP50-95", OUTPUT_DIR / "weather_map50_95_bar.png", "mAP50-95")
    create_bar_chart(weather_df.sort_values("precision", ascending=False), "weather", "precision", "ACDC weather-wise comparison: precision", OUTPUT_DIR / "weather_precision_bar.png", "precision")
    create_bar_chart(weather_df.sort_values("recall", ascending=False), "weather", "recall", "ACDC weather-wise comparison: recall", OUTPUT_DIR / "weather_recall_bar.png", "recall")

    training_curve_created = create_training_curve(RUN_DIR / "results.csv", OUTPUT_DIR / "training_curve_from_results_csv.png")

    rank_map = infer_weather_rank(weather_df, "mAP50_95")
    best_weather = rank_map[0]
    worst_weather = rank_map[-1]

    bdd_map50 = overall_df.loc[overall_df["dataset"] == "BDD100K_val", "mAP50"].iloc[0]
    bdd_map5095 = overall_df.loc[overall_df["dataset"] == "BDD100K_val", "mAP50_95"].iloc[0]
    acdc_map50 = overall_df.loc[overall_df["dataset"] == "ACDC_val_gt", "mAP50"].iloc[0]
    acdc_map5095 = overall_df.loc[overall_df["dataset"] == "ACDC_val_gt", "mAP50_95"].iloc[0]

    overall_drop_map50 = bdd_map50 - acdc_map50
    overall_drop_map5095 = bdd_map5095 - acdc_map5095
    overall_rel_map50 = pct_drop(bdd_map50, acdc_map50)
    overall_rel_map5095 = pct_drop(bdd_map5095, acdc_map5095)

    report_lines = []
    report_lines.append("# BDD100K -> ACDC YOLOv8n Evaluation Report")
    report_lines.append("")
    report_lines.append("## 1. Experiment scope")
    report_lines.append("")
    report_lines.append("- Detector: YOLOv8n")
    report_lines.append("- Source-domain training/validation: BDD100K 8-class object detection")
    report_lines.append("- Target-domain evaluation: ACDC adverse-weather validation")
    report_lines.append("- Classes: person, rider, car, truck, bus, train, motorcycle, bicycle")
    report_lines.append("")
    report_lines.append("## 2. Executive summary")
    report_lines.append("")
    report_lines.append(f"- In-domain BDD100K validation reached **mAP50 = {bdd_map50:.4f}** and **mAP50-95 = {bdd_map5095:.4f}**.")
    report_lines.append(f"- Cross-domain ACDC validation reached **mAP50 = {acdc_map50:.4f}** and **mAP50-95 = {acdc_map5095:.4f}**.")
    report_lines.append(f"- Absolute cross-domain drop was **{overall_drop_map50:.4f}** in mAP50 and **{overall_drop_map5095:.4f}** in mAP50-95.")
    report_lines.append(f"- Relative cross-domain drop was **{overall_rel_map50:.2f}%** in mAP50 and **{overall_rel_map5095:.2f}%** in mAP50-95.")
    report_lines.append(f"- Best adverse-weather domain by mAP50-95 was **{best_weather}**; worst was **{worst_weather}**.")
    report_lines.append("")
    report_lines.append("## 3. Overall metrics table")
    report_lines.append("")
    report_lines.append(overall_df.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 4. Weather-wise metrics table")
    report_lines.append("")
    report_lines.append(weather_df.sort_values('mAP50_95', ascending=False).to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 5. Interpretation")
    report_lines.append("")
    report_lines.append("### Overall domain shift")
    report_lines.append("")
    report_lines.append("The model generalized imperfectly from clear-weather BDD100K to adverse-weather ACDC. This indicates a meaningful domain-shift penalty when visual conditions change.")
    report_lines.append("")
    report_lines.append("### Weather-wise behavior")
    report_lines.append("")
    for _, row in weather_df.sort_values("mAP50_95", ascending=False).iterrows():
        report_lines.append(
            f"- **{row['weather']}**: mAP50 = {row['mAP50']:.4f}, mAP50-95 = {row['mAP50_95']:.4f}, precision = {row['precision']:.4f}, recall = {row['recall']:.4f}."
        )
    report_lines.append("")
    report_lines.append(f"By mAP50-95, the ranking was: **{' > '.join(rank_map)}**.")
    report_lines.append("")
    report_lines.append("### Practical implications")
    report_lines.append("")
    report_lines.append("For the ADAS motivation of the project, the results imply that a detector trained on normal conditions can lose substantial reliability under adverse weather, especially in the weakest-performing weather domains. This motivates weather-aware training, domain adaptation, or more robust multi-condition data collection.")
    report_lines.append("")
    report_lines.append("## 6. Available generated figures")
    report_lines.append("")
    figure_names = [
        "overall_map50_bar.png",
        "overall_map50_95_bar.png",
        "weather_map50_bar.png",
        "weather_map50_95_bar.png",
        "weather_precision_bar.png",
        "weather_recall_bar.png",
    ]
    if training_curve_created:
        figure_names.append("training_curve_from_results_csv.png")
    for name in figure_names:
        report_lines.append(f"- `{name}`")
    report_lines.append("")
    report_lines.append("## 7. Suggested report narrative")
    report_lines.append("")
    report_lines.append("A clear-weather-trained YOLOv8n detector achieved stronger in-domain performance on BDD100K than on the adverse-weather ACDC benchmark. The overall reduction in mAP confirms a measurable domain-shift effect. Weather-wise analysis further showed that performance degradation was not uniform across fog, night, rain, and snow, indicating that distinct adverse conditions stress different visual cues used by the detector.")

    report_md = "\n".join(report_lines)
    with open(OUTPUT_DIR / "final_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Saved package to: {OUTPUT_DIR}")
    print("Files created:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    summarize()

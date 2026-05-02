# -*- coding: utf-8 -*-
"""
Step L6: Visualize ACDC lane robustness results from weather summary CSV.

What it does
------------
1) Reads acdc_lane_weather_summary_all.csv
2) Builds a report-ready table CSV
3) Creates bar charts for:
   - detection rate
   - no-detection rate
   - average lane detections per image
   - average confidence
4) Creates a compact markdown summary

Author: mahek
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================
RUN_DIR = Path(r"C:\UDMercy\Semester 1\Project\lane_runs\yolov8n_bdd100k_lane_seg2")
SUMMARY_CSV = RUN_DIR / "acdc_lane_weather_summary_all.csv"
OUTPUT_DIR = RUN_DIR / "step_L6_visualizations"
# ============================================================


def make_detection_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["detection_rate"] = 1.0 - df["no_detection_rate"]
    df["detection_rate_pct"] = df["detection_rate"] * 100.0
    df["no_detection_rate_pct"] = df["no_detection_rate"] * 100.0
    return df


def save_table(df: pd.DataFrame):
    out_df = df[[
        "weather",
        "num_images",
        "images_with_detection",
        "images_without_detection",
        "detection_rate_pct",
        "no_detection_rate_pct",
        "avg_lane_detections_per_image",
        "avg_confidence",
        "max_lane_detections_in_an_image",
    ]].copy()

    out_df.rename(columns={
        "weather": "Weather",
        "num_images": "Images",
        "images_with_detection": "Images with Detection",
        "images_without_detection": "Images without Detection",
        "detection_rate_pct": "Detection Rate (%)",
        "no_detection_rate_pct": "No Detection Rate (%)",
        "avg_lane_detections_per_image": "Avg Lane Detections / Image",
        "avg_confidence": "Avg Confidence",
        "max_lane_detections_in_an_image": "Max Lane Detections in an Image",
    }, inplace=True)

    out_csv = OUTPUT_DIR / "lane_weather_results_table.csv"
    out_df.to_csv(out_csv, index=False)
    return out_df, out_csv


def make_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, out_path: Path, sort_desc=True):
    plot_df = df.sort_values(y, ascending=not sort_desc)
    plt.figure(figsize=(8, 5))
    plt.bar(plot_df[x], plot_df[y])
    plt.title(title)
    plt.xlabel("Weather")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_markdown(df: pd.DataFrame):
    best_conf = df.sort_values("avg_confidence", ascending=False).iloc[0]["weather"]
    worst_conf = df.sort_values("avg_confidence", ascending=True).iloc[0]["weather"]
    best_det = df.sort_values("detection_rate_pct", ascending=False).iloc[0]["weather"]
    worst_det = df.sort_values("detection_rate_pct", ascending=True).iloc[0]["weather"]
    best_lane = df.sort_values("avg_lane_detections_per_image", ascending=False).iloc[0]["weather"]
    worst_lane = df.sort_values("avg_lane_detections_per_image", ascending=True).iloc[0]["weather"]

    lines = []
    lines.append("# Step L6 Lane Visualization Summary")
    lines.append("")
    lines.append("## Main findings")
    lines.append("")
    lines.append(f"- Best detection-rate weather: **{best_det}**")
    lines.append(f"- Worst detection-rate weather: **{worst_det}**")
    lines.append(f"- Highest average confidence: **{best_conf}**")
    lines.append(f"- Lowest average confidence: **{worst_conf}**")
    lines.append(f"- Highest average lane detections/image: **{best_lane}**")
    lines.append(f"- Lowest average lane detections/image: **{worst_lane}**")
    lines.append("")
    lines.append("## Report-ready interpretation")
    lines.append("")
    lines.append(
        "The adverse-weather lane robustness study shows that lane-related segmentation "
        "performance is not uniform across weather conditions. Detection continuity and "
        "confidence remain strongest in fog and night, while rain and especially snow "
        "reduce both lane visibility and prediction stability. This supports the project "
        "motivation that severe winter conditions introduce the highest risk for lane-following assistance."
    )
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    for p in sorted(OUTPUT_DIR.iterdir()):
        lines.append(f"- `{p.name}`")

    md_path = OUTPUT_DIR / "step_L6_lane_visualization_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)
    df = make_detection_rate(df)

    table_df, table_csv = save_table(df)

    make_bar(
        df, "weather", "detection_rate_pct",
        "ACDC lane robustness: detection rate by weather",
        "Detection Rate (%)",
        OUTPUT_DIR / "lane_detection_rate_bar.png",
        sort_desc=True,
    )

    make_bar(
        df, "weather", "no_detection_rate_pct",
        "ACDC lane robustness: no-detection rate by weather",
        "No Detection Rate (%)",
        OUTPUT_DIR / "lane_no_detection_rate_bar.png",
        sort_desc=True,
    )

    make_bar(
        df, "weather", "avg_lane_detections_per_image",
        "ACDC lane robustness: avg lane detections per image",
        "Avg Lane Detections / Image",
        OUTPUT_DIR / "lane_avg_detections_bar.png",
        sort_desc=True,
    )

    make_bar(
        df, "weather", "avg_confidence",
        "ACDC lane robustness: avg confidence by weather",
        "Average Confidence",
        OUTPUT_DIR / "lane_avg_confidence_bar.png",
        sort_desc=True,
    )

    md_path = write_markdown(df)

    print("Step L6 visualization package created at:")
    print(OUTPUT_DIR)
    print("\nFiles:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(" -", p.name)
    print("\nMain table preview:")
    print(table_df.to_string(index=False))
    print(f"\nMarkdown summary: {md_path}")


if __name__ == "__main__":
    main()

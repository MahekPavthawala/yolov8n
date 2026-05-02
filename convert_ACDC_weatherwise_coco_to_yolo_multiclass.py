# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:25:58 2026

Create weather-wise ACDC YOLO validation datasets:
fog, night, rain, snow

Each weather becomes an independent YOLO validation dataset.

Author: mahek
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================
IMAGE_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\rgb_anon\rgb_anon")
ANNOTATION_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\gt_detection\gt_detection")
OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\acdc_yolo_weatherwise_8cls")
# ============================================================


WEATHERS = ["fog", "night", "rain", "snow"]

CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

CATEGORY_ID_MAP = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    31: 5,  # train
    32: 6,  # motorcycle
    33: 7,  # bicycle
}


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h


def clamp(v, min_v=0.0, max_v=1.0):
    return max(min_v, min(v, max_v))


def flatten_rel_path(rel_path):
    return "_".join(rel_path.parts)


def make_stats():
    return {
        "images": 0,
        "labels": 0,
        "objects": 0,
        "missing": 0,
        "class_counts": [0] * len(CLASS_NAMES),
    }


def write_yaml(weather_root, weather):
    yaml_path = weather_root / f"acdc_{weather}.yaml"

    yaml_text = f"""path: {weather_root.as_posix()}
train: images/val
val: images/val

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

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"YAML written: {yaml_path}")


def process_weather(weather):
    print(f"\n================ {weather.upper()} =================\n")

    json_path = ANNOTATION_ROOT / weather / f"instancesonly_{weather}_val_gt_detection.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON: {json_path}")

    weather_root = OUTPUT_ROOT / weather
    img_out = weather_root / "images" / "val"
    lbl_out = weather_root / "labels" / "val"

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    image_id_to_info = {img["id"]: img for img in images}

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    stats = make_stats()

    for image_id, img_info in image_id_to_info.items():
        rel_path = Path(img_info["file_name"])
        src_img = IMAGE_ROOT / rel_path

        if not src_img.exists():
            print(f"[WARNING] Missing image: {src_img}")
            stats["missing"] += 1
            continue

        dst_name = flatten_rel_path(rel_path)
        dst_img = img_out / dst_name

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        stats["images"] += 1

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_lines = []

        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            cat_id = ann["category_id"]
            if cat_id not in CATEGORY_ID_MAP:
                continue

            bbox = ann["bbox"]
            x_center, y_center, w, h = coco_to_yolo_bbox(bbox, img_w, img_h)

            x_center = clamp(x_center)
            y_center = clamp(y_center)
            w = clamp(w)
            h = clamp(h)

            if w <= 0 or h <= 0:
                continue

            cls = CATEGORY_ID_MAP[cat_id]
            label_lines.append(
                f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

            stats["objects"] += 1
            stats["class_counts"][cls] += 1

        label_path = lbl_out / f"{dst_img.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        stats["labels"] += 1

    write_yaml(weather_root, weather)

    print(f"Images : {stats['images']}")
    print(f"Labels : {stats['labels']}")
    print(f"Objects: {stats['objects']}")
    print(f"Missing: {stats['missing']}")

    print("Class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name:<12} {stats['class_counts'][i]}")

    return stats


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for weather in WEATHERS:
        stats = process_weather(weather)
        all_stats[weather] = stats

    print("\n================ DONE =================")
    print(f"Weather-wise ACDC datasets saved at:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
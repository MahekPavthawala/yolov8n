# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:44:18 2026

@author: mahek
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# UPDATE THESE PATHS FOR YOUR PC
# ============================================================
IMAGE_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\rgb_anon\rgb_anon")
TRAIN_JSON = Path(r"C:\UDMercy\Semester 1\Project\ACDC\gt_detection\gt_detection\fog\instancesonly_fog_train_gt_detection.json")
VAL_JSON = Path(r"C:\UDMercy\Semester 1\Project\ACDC\gt_detection\gt_detection\fog\instancesonly_fog_val_gt_detection.json")
OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\yolo_acdc_fog")
# ============================================================

# Original ACDC IDs -> YOLO class IDs
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


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox  # COCO format

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h


def clamp(v, min_v=0.0, max_v=1.0):
    return max(min_v, min(v, max_v))


def process_split(json_path, split_name):
    print(f"\nProcessing {split_name}: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    image_id_to_info = {img["id"]: img for img in images}

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    out_img_dir = OUTPUT_ROOT / "images" / split_name
    out_lbl_dir = OUTPUT_ROOT / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    written = 0

    for image_id, img_info in image_id_to_info.items():
        rel_path = Path(img_info["file_name"])   # e.g. fog/train/GP010475/xxx.png
        src_img = IMAGE_ROOT / rel_path

        if not src_img.exists():
            print(f"[WARNING] Missing image: {src_img}")
            missing += 1
            continue

        # Make filename unique and safe
        dst_name = "_".join(rel_path.parts)
        dst_img = out_img_dir / dst_name
        shutil.copy2(src_img, dst_img)
        copied += 1

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_lines = []
        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            cat_id = ann["category_id"]
            if cat_id not in CATEGORY_ID_MAP:
                continue

            class_id = CATEGORY_ID_MAP[cat_id]
            bbox = ann["bbox"]

            x_center, y_center, w, h = coco_to_yolo_bbox(bbox, img_w, img_h)

            x_center = clamp(x_center)
            y_center = clamp(y_center)
            w = clamp(w)
            h = clamp(h)

            if w <= 0 or h <= 0:
                continue

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )

        label_file = out_lbl_dir / f"{dst_img.stem}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        written += 1

    print(f"Finished {split_name}")
    print(f"Images copied : {copied}")
    print(f"Labels written: {written}")
    print(f"Missing images: {missing}")


def write_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
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
        f.write(text)

    print(f"\nCreated YAML: {yaml_path}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    process_split(TRAIN_JSON, "train")
    process_split(VAL_JSON, "val")
    write_yaml()

    print("\nDone.")
    print(f"YOLO dataset saved at: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
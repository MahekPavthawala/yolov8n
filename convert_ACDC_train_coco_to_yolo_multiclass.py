# -*- coding: utf-8 -*-
"""
Convert ACDC train GT detection annotations to YOLO 8-class format.

Output:
acdc_yolo_8cls/
└── train_gt/
    ├── images/train
    ├── labels/train
    └── acdc_train_gt.yaml

Author: mahek
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# UPDATE PATHS IF NEEDED
# ============================================================
IMAGE_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\rgb_anon\rgb_anon")
ANNOTATION_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\gt_detection\gt_detection")
OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\acdc_yolo_8cls\train_gt")
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

# Same 8-class mapping used in your current ACDC object pipeline
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
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w = w / img_w
    h = h / img_h
    return x_center, y_center, w, h


def clamp(v, min_v=0.0, max_v=1.0):
    return max(min_v, min(v, max_v))


def flatten_rel_path(rel_path):
    return "_".join(rel_path.parts)


def make_stats():
    return {
        "images_copied": 0,
        "labels_written": 0,
        "missing_images": 0,
        "objects_kept": 0,
        "empty_label_files": 0,
        "class_counts": [0] * len(CLASS_NAMES),
    }


def process_json(json_path, split_name, dataset_root, stats):
    print(f"\nProcessing {split_name}: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    image_id_to_info = {img["id"]: img for img in images}

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    out_img_dir = dataset_root / "images" / split_name
    out_lbl_dir = dataset_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    written = 0
    missing = 0
    objects_kept = 0
    empty_files = 0
    local_class_counts = [0] * len(CLASS_NAMES)

    for image_id, img_info in image_id_to_info.items():
        rel_path = Path(img_info["file_name"])
        src_img = IMAGE_ROOT / rel_path

        if not src_img.exists():
            print(f"[WARNING] Missing image: {src_img}")
            missing += 1
            continue

        dst_name = flatten_rel_path(rel_path)
        dst_img = out_img_dir / dst_name

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        copied += 1

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_lines = []

        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            cat_id = ann.get("category_id")
            if cat_id not in CATEGORY_ID_MAP:
                continue

            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            x_center, y_center, w, h = coco_to_yolo_bbox(bbox, img_w, img_h)

            x_center = clamp(x_center)
            y_center = clamp(y_center)
            w = clamp(w)
            h = clamp(h)

            if w <= 0 or h <= 0:
                continue

            cls = CATEGORY_ID_MAP[cat_id]
            label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            objects_kept += 1
            local_class_counts[cls] += 1

        label_path = out_lbl_dir / f"{dst_img.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        written += 1
        if len(label_lines) == 0:
            empty_files += 1

    print(f"Finished {json_path.name}")
    print(f"  Images copied    : {copied}")
    print(f"  Labels written   : {written}")
    print(f"  Missing images   : {missing}")
    print(f"  Objects kept     : {objects_kept}")
    print(f"  Empty label files: {empty_files}")

    print("  Class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i}: {name:<12} {local_class_counts[i]}")

    stats["images_copied"] += copied
    stats["labels_written"] += written
    stats["missing_images"] += missing
    stats["objects_kept"] += objects_kept
    stats["empty_label_files"] += empty_files

    for i in range(len(CLASS_NAMES)):
        stats["class_counts"][i] += local_class_counts[i]


def write_yaml():
    yaml_file = OUTPUT_ROOT / "acdc_train_gt.yaml"
    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/train

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
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\nYAML written to: {yaml_file}")


def print_summary(stats):
    print(f"\n================ TRAIN_GT SUMMARY ================")
    print(f"Images copied    : {stats['images_copied']}")
    print(f"Labels written   : {stats['labels_written']}")
    print(f"Missing images   : {stats['missing_images']}")
    print(f"Objects kept     : {stats['objects_kept']}")
    print(f"Empty label files: {stats['empty_label_files']}")
    print("Class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name:<12} {stats['class_counts'][i]}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stats = make_stats()

    for weather in WEATHERS:
        json_path = ANNOTATION_ROOT / weather / f"instancesonly_{weather}_train_gt_detection.json"

        if json_path.exists():
            process_json(
                json_path=json_path,
                split_name="train",
                dataset_root=OUTPUT_ROOT,
                stats=stats,
            )
        else:
            print(f"[WARNING] Missing train GT JSON: {json_path}")

    write_yaml()
    print_summary(stats)
    print(f"\nYOLO ACDC train_gt dataset ready at:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:04:55 2026

@author: mahek
"""

import json
import shutil
from pathlib import Path
from PIL import Image

BDD_ROOT = Path(r"C:\UDMercy\Semester 1\Project\BDD100K")
IMAGE_ROOT = BDD_ROOT / "images" / "100k"
LABEL_ROOT = BDD_ROOT / "labels" / "100k"

OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\bdd100k_yolo_8cls")

CLASS_MAP = {
    "person": 0,
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "motorcycle": 6,
    "bike": 7,
    "bicycle": 7,
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


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def convert_split(split):
    img_dir = IMAGE_ROOT / split
    json_dir = LABEL_ROOT / split

    out_img = OUTPUT_ROOT / "images" / split
    out_lbl = OUTPUT_ROOT / "labels" / split

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    count_images = 0
    count_objects = 0
    class_counts = [0] * len(CLASS_NAMES)
    missing_json = 0

    for img_path in img_dir.glob("*.jpg"):
        json_path = json_dir / f"{img_path.stem}.json"

        if not json_path.exists():
            missing_json += 1
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with Image.open(img_path) as im:
            w, h = im.size

        yolo_lines = []

        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category")

                if category not in CLASS_MAP:
                    continue

                box = obj.get("box2d")
                if box is None:
                    continue

                x1 = box["x1"]
                y1 = box["y1"]
                x2 = box["x2"]
                y2 = box["y2"]

                if x2 <= x1 or y2 <= y1:
                    continue

                xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    continue

                cls = CLASS_MAP[category]
                yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                count_objects += 1
                class_counts[cls] += 1

        if len(yolo_lines) == 0:
            continue

        count_images += 1

        target_img = out_img / img_path.name
        if not target_img.exists():
            shutil.copy2(img_path, target_img)

        target_label = out_lbl / f"{img_path.stem}.txt"
        with open(target_label, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print(f"\n{split}:")
    print(f"  Images converted : {count_images}")
    print(f"  Objects kept     : {count_objects}")
    print(f"  Missing JSON     : {missing_json}")
    print("  Class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i}: {name:<12} {class_counts[i]}")


def write_yaml():
    yaml_file = OUTPUT_ROOT / "bdd100k.yaml"
    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
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
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\nYAML written to: {yaml_file}")


def main():
    convert_split("train")
    convert_split("val")
    write_yaml()
    print("\nYOLO BDD100K 8-class dataset ready.")


if __name__ == "__main__":
    main()
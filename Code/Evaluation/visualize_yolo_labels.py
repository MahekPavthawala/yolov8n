# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:02:14 2026

@author: mahek
"""

import random
from pathlib import Path
import cv2

DATASET_ROOT = Path(r"C:\UDMercy\Semester 1\Project\ACDC\yolo_acdc_fog")
IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"
OUTPUT_DIR = DATASET_ROOT / "debug_vis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

image_files = list(IMAGE_DIR.glob("*.png"))
sample_images = random.sample(image_files, min(10, len(image_files)))

for img_path in sample_images:
    label_path = LABEL_DIR / f"{img_path.stem}.txt"
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"Could not read {img_path}")
        continue

    h, w = img.shape[:2]

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:])

            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = CLASS_NAMES[class_id]
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), img)

print(f"Saved visualizations to: {OUTPUT_DIR}")
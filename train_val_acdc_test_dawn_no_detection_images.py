# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:03:23 2026

@author: mahek
"""

from pathlib import Path

image_dir = Path(r"C:\UDMercy\Semester 1\Project\dawn_yolo_8cls\images\test")
pred_label_dir = Path(r"C:\UDMercy\Semester 1\Project\yolo_runs_acdc_only\yolov8n_acdc_train_acdcval_dawn_test_predict_dawn\labels")

image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_files.extend(image_dir.glob(ext))

txt_files = list(pred_label_dir.glob("*.txt"))

image_stems = {img.stem for img in image_files}
txt_stems = {txt.stem for txt in txt_files}

detected_images = image_stems & txt_stems
no_detection_images = image_stems - txt_stems

print("Total images:", len(image_stems))
print("Detected images:", len(detected_images))
print("No-detection images:", len(no_detection_images))
print("No-detection rate:", len(no_detection_images) / len(image_stems) if image_stems else 0)

# Optional: print a few no-detection image names
print("\nSample no-detection images:")
for name in sorted(list(no_detection_images))[:20]:
    print(name)
# -*- coding: utf-8 -*-
"""
Convert binary lane masks into YOLOv8 segmentation labels.

Input:
bdd100k_lane_seg/
├── images/train
├── images/val
├── masks/train
└── masks/val

Output:
bdd100k_lane_yolo_seg/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── bdd100k_lane_seg.yaml

YOLO seg label format:
class_id x1 y1 x2 y2 x3 y3 ...

Author: mahek
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# UPDATE PATHS IF NEEDED
# ============================================================
SRC_ROOT = Path(r"C:\UDMercy\Semester 1\Project\bdd100k_lane_seg")
DST_ROOT = Path(r"C:\UDMercy\Semester 1\Project\bdd100k_lane_yolo_seg")
# ============================================================


CLASS_ID = 0

# Small contours below this area are ignored
MIN_CONTOUR_AREA = 20

# Polygon simplification factor:
# approx = epsilon_ratio * contour_perimeter
EPSILON_RATIO = 0.002

# Copy images into YOLO-seg dataset structure
COPY_IMAGES = True


def make_dirs():
    for split in ["train", "val"]:
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def normalize_polygon(points, width, height):
    """
    Convert pixel coordinates into YOLO-normalized coordinates.
    """
    norm = []
    for x, y in points:
        xn = max(0.0, min(1.0, x / width))
        yn = max(0.0, min(1.0, y / height))
        norm.extend([xn, yn])
    return norm


def contour_to_polygon(contour):
    """
    contour shape from OpenCV: (N, 1, 2)
    returns list of (x, y)
    """
    pts = contour.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in pts]


def mask_to_yolo_lines(mask_path):
    """
    Read one binary mask and convert all valid contours to YOLO-seg label lines.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    h, w = mask.shape

    # Ensure binary
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    kept_contours = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        peri = cv2.arcLength(contour, closed=True)
        epsilon = EPSILON_RATIO * peri
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        polygon = contour_to_polygon(approx)

        # Need at least 3 points => 6 coordinates
        if len(polygon) < 3:
            continue

        norm_coords = normalize_polygon(polygon, w, h)

        if len(norm_coords) < 6:
            continue

        line = str(CLASS_ID) + " " + " ".join(f"{v:.6f}" for v in norm_coords)
        lines.append(line)
        kept_contours += 1

    return lines, kept_contours


def write_yaml():
    yaml_path = DST_ROOT / "bdd100k_lane_seg.yaml"
    yaml_text = f"""path: {DST_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: lane
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"YAML written: {yaml_path}")


def process_split(split):
    print(f"\n================ {split.upper()} =================\n")

    src_img_dir = SRC_ROOT / "images" / split
    src_msk_dir = SRC_ROOT / "masks" / split

    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split

    if not src_img_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {src_img_dir}")
    if not src_msk_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {src_msk_dir}")

    image_files = sorted(
        list(src_img_dir.glob("*.jpg")) +
        list(src_img_dir.glob("*.jpeg")) +
        list(src_img_dir.glob("*.png"))
    )

    total_images = 0
    total_labels_written = 0
    total_contours_kept = 0
    empty_labels = 0
    missing_masks = 0

    for img_path in image_files:
        total_images += 1

        mask_path = src_msk_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"[WARNING] Missing mask for {img_path.name}")
            missing_masks += 1
            continue

        yolo_lines, kept_contours = mask_to_yolo_lines(mask_path)

        label_path = dst_lbl_dir / f"{img_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        total_labels_written += 1
        total_contours_kept += kept_contours

        if len(yolo_lines) == 0:
            empty_labels += 1

        if COPY_IMAGES:
            dst_img_path = dst_img_dir / img_path.name
            if not dst_img_path.exists():
                shutil.copy2(img_path, dst_img_path)

        if total_images % 1000 == 0:
            print(f"Processed {total_images} images...")

    print(f"Images processed      : {total_images}")
    print(f"Labels written        : {total_labels_written}")
    print(f"Contours kept         : {total_contours_kept}")
    print(f"Empty label files     : {empty_labels}")
    print(f"Missing masks         : {missing_masks}")


def main():
    make_dirs()

    process_split("train")
    process_split("val")

    write_yaml()

    print(f"\nDone. Output saved to:\n{DST_ROOT}")


if __name__ == "__main__":
    main()
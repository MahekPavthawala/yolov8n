# -*- coding: utf-8 -*-
"""
Convert BDD100K per-image JSON lane annotations into binary lane masks.

This version matches the user's actual BDD100K schema:
- per-image JSON files
- annotations inside data["frames"][...]["objects"]
- lane categories like "lane/single white"
- poly2d points stored directly as [x, y, "L"/"C"]

Output:
bdd100k_lane_seg/
├── images/train
├── images/val
├── masks/train
├── masks/val

Mask encoding:
- 0   = background
- 255 = lane marking

Author: mahek
"""

import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw


# ============================================================
# UPDATE THESE PATHS TO MATCH YOUR LOCAL DATASET
# ============================================================
BDD_ROOT = Path(r"C:\UDMercy\Semester 1\Project\BDD100K")

IMAGE_ROOT = BDD_ROOT / "images" / "100k"
LABEL_ROOT = BDD_ROOT / "labels" / "100k"

IMAGE_TRAIN_DIR = IMAGE_ROOT / "train"
IMAGE_VAL_DIR   = IMAGE_ROOT / "val"

LABEL_TRAIN_DIR = LABEL_ROOT / "train"
LABEL_VAL_DIR   = LABEL_ROOT / "val"

OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\bdd100k_lane_seg")
# ============================================================


# Keep only lane-related categories
LANE_CATEGORIES = {
    "lane/single white",
    "lane/single yellow",
    "lane/double white",
    "lane/double yellow",
    "lane/single other",
    "lane/double other",
    "lane/road curb",
    "lane/crosswalk",
}

# Optional: include drivable-area polygons too
INCLUDE_DRIVABLE = False
DRIVABLE_CATEGORIES = {
    "area/drivable",
    "area/alternative",
}

LINE_WIDTH = 8
COPY_IMAGES = True


def make_dirs():
    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "masks" / split).mkdir(parents=True, exist_ok=True)


def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_lane_object(obj):
    category = str(obj.get("category", "")).strip().lower()
    return category in LANE_CATEGORIES


def is_drivable_object(obj):
    category = str(obj.get("category", "")).strip().lower()
    return category in DRIVABLE_CATEGORIES


def extract_poly_points_from_object(obj):
    """
    In the user's JSON, poly2d looks like:
    "poly2d": [
        [x, y, "C"],
        [x, y, "C"],
        [x, y, "L"]
    ]

    We convert it into [(x1,y1), (x2,y2), ...]
    """
    poly = obj.get("poly2d", [])
    pts = []

    if not isinstance(poly, list):
        return pts

    for p in poly:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
            pts.append((int(round(x)), int(round(y))))

    return pts


def create_blank_mask(width, height):
    return Image.new("L", (width, height), 0)


def draw_lane_line(mask, pts, line_width):
    if len(pts) < 2:
        return
    draw = ImageDraw.Draw(mask)
    draw.line(pts, fill=255, width=line_width)


def draw_drivable_polygon(mask, pts):
    if len(pts) < 3:
        return
    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, fill=255)


def process_split(split_name, image_dir, label_dir):
    print(f"\n================ {split_name.upper()} =================\n")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    out_img_dir = OUTPUT_ROOT / "images" / split_name
    out_msk_dir = OUTPUT_ROOT / "masks" / split_name

    json_files = sorted(label_dir.glob("*.json"))

    total_json = 0
    images_found = 0
    masks_written = 0
    missing_images = 0
    json_errors = 0

    images_with_lane = 0
    total_lane_instances = 0
    total_drivable_instances = 0

    category_counts = {}

    for json_path in json_files:
        total_json += 1

        try:
            data = read_json(json_path)
        except Exception as e:
            print(f"[WARNING] Could not read {json_path.name}: {e}")
            json_errors += 1
            continue

        img_path = image_dir / f"{json_path.stem}.jpg"
        if not img_path.exists():
            # fallback options if needed
            alt_png = image_dir / f"{json_path.stem}.png"
            alt_jpeg = image_dir / f"{json_path.stem}.jpeg"
            if alt_png.exists():
                img_path = alt_png
            elif alt_jpeg.exists():
                img_path = alt_jpeg
            else:
                print(f"[WARNING] No matching image found for: {json_path.name}")
                missing_images += 1
                continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"[WARNING] Could not open image {img_path.name}: {e}")
            continue

        mask = create_blank_mask(width, height)

        lane_count_this_image = 0

        frames = data.get("frames", [])
        for frame in frames:
            objects = frame.get("objects", [])
            if not isinstance(objects, list):
                continue

            for obj in objects:
                if not isinstance(obj, dict):
                    continue

                category = str(obj.get("category", "")).strip().lower()
                category_counts[category] = category_counts.get(category, 0) + 1

                pts = extract_poly_points_from_object(obj)
                if len(pts) == 0:
                    continue

                if is_lane_object(obj):
                    draw_lane_line(mask, pts, LINE_WIDTH)
                    lane_count_this_image += 1
                    total_lane_instances += 1

                elif INCLUDE_DRIVABLE and is_drivable_object(obj):
                    draw_drivable_polygon(mask, pts)
                    total_drivable_instances += 1

        if lane_count_this_image > 0:
            images_with_lane += 1

        mask_path = out_msk_dir / f"{img_path.stem}.png"
        mask.save(mask_path)
        masks_written += 1

        if COPY_IMAGES:
            out_img_path = out_img_dir / img_path.name
            if not out_img_path.exists():
                shutil.copy2(img_path, out_img_path)

        images_found += 1

        if images_found % 1000 == 0:
            print(f"Processed {images_found} images...")

    print(f"JSON files found       : {total_json}")
    print(f"Images matched         : {images_found}")
    print(f"Masks written          : {masks_written}")
    print(f"Missing images         : {missing_images}")
    print(f"JSON read errors       : {json_errors}")
    print(f"Images with lane marks : {images_with_lane}")
    print(f"Total lane instances   : {total_lane_instances}")
    if INCLUDE_DRIVABLE:
        print(f"Total drivable objects : {total_drivable_instances}")

    print("\nTop discovered categories:")
    top_items = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for cat, cnt in top_items:
        print(f"  {cat:<24} {cnt}")


def write_dataset_readme():
    readme_path = OUTPUT_ROOT / "README_lane_seg.txt"
    text = f"""BDD100K Lane Segmentation Dataset

Mask encoding:
- 0   = background
- 255 = lane marking

Lane categories kept:
{sorted(LANE_CATEGORIES)}

Include drivable area: {INCLUDE_DRIVABLE}
Rasterized line width: {LINE_WIDTH}
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    make_dirs()

    process_split("train", IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR)
    process_split("val", IMAGE_VAL_DIR, LABEL_VAL_DIR)

    write_dataset_readme()

    print(f"\nDone. Output saved to:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
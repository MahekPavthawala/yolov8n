# -*- coding: utf-8 -*-
"""
Convert DAWN dataset (nested ZIP + Pascal VOC XML) into YOLO 8-class format.

Input:
- outer DAWN zip file downloaded from Mendeley

Output:
dawn_yolo_8cls/
├── images/test
├── labels/test
└── dawn_8cls.yaml

Notes
-----
- DAWN download contains nested zips: Fog.zip, Rain.zip, Snow.zip, Sand.zip
- Each inner zip contains:
    - images
    - Pascal VOC XML annotations
    - YOLO_darknet txt labels
- This script uses XML annotations so we can control the class mapping cleanly.
- DAWN does not appear to contain the "rider" class, so that class will simply be unused.

Author: mahek
"""

from pathlib import Path
import zipfile
import io
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict


# ============================================================
# UPDATE PATHS IF NEEDED
# ============================================================
DAWN_ZIP = Path(r"C:\UDMercy\Semester 1\Project\766ygrbt8y-3.zip")
OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\dawn_yolo_8cls")
# ============================================================


CLASS_NAMES = [
    "person",      # 0
    "rider",       # 1
    "car",         # 2
    "truck",       # 3
    "bus",         # 4
    "train",       # 5
    "motorcycle",  # 6
    "bicycle",     # 7
]

# DAWN class-name -> your 8-class mapping
CLASS_MAP = {
    "person": 0,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "motorbike": 6,
    "bicycle": 7,
    "bike": 7,
}

INNER_ZIPS = {
    "fog": "DAWN/Fog.zip",
    "rain": "DAWN/Rain.zip",
    "snow": "DAWN/Snow.zip",
    "sand": "DAWN/Sand.zip",
}


def clamp(v, min_v=0.0, max_v=1.0):
    return max(min_v, min(v, max_v))


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    xc = ((xmin + xmax) / 2.0) / img_w
    yc = ((ymin + ymax) / 2.0) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return xc, yc, bw, bh


def parse_voc_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)

    filename = root.findtext("filename")
    size = root.find("size")
    if size is None:
        return None

    img_w = int(float(size.findtext("width", "0")))
    img_h = int(float(size.findtext("height", "0")))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return {
        "filename": filename,
        "width": img_w,
        "height": img_h,
        "objects": objects,
    }


def write_yaml():
    yaml_path = OUTPUT_ROOT / "dawn_8cls.yaml"
    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
test: images/test

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
    print(f"\nYAML written to: {yaml_path}")


def make_dirs():
    (OUTPUT_ROOT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / "test").mkdir(parents=True, exist_ok=True)


def convert():
    make_dirs()

    out_img_dir = OUTPUT_ROOT / "images" / "test"
    out_lbl_dir = OUTPUT_ROOT / "labels" / "test"

    stats = {
        "images_written": 0,
        "labels_written": 0,
        "objects_kept": 0,
        "xml_files_seen": 0,
        "unknown_classes": defaultdict(int),
        "class_counts": [0] * len(CLASS_NAMES),
        "weather_counts": defaultdict(int),
    }

    with zipfile.ZipFile(DAWN_ZIP, "r") as outer_zip:
        for weather, inner_zip_name in INNER_ZIPS.items():
            if inner_zip_name not in outer_zip.namelist():
                print(f"[WARNING] Missing inner zip: {inner_zip_name}")
                continue

            print(f"\n================ {weather.upper()} =================")

            inner_bytes = outer_zip.read(inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_zip:
                names = inner_zip.namelist()

                xml_files = [n for n in names if "PASCAL_VOC/" in n and n.lower().endswith(".xml")]
                print(f"XML files found: {len(xml_files)}")

                for xml_name in xml_files:
                    stats["xml_files_seen"] += 1

                    parsed = parse_voc_xml(inner_zip.read(xml_name))
                    if parsed is None:
                        continue

                    img_filename = parsed["filename"]
                    if not img_filename:
                        continue

                    # Find the matching image inside the inner zip
                    # Usually it's directly under weather folder, e.g. Fog/haze-027.jpg
                    image_candidates = [
                        f"{weather.capitalize()}/{img_filename}",
                        f"{weather}/{img_filename}",
                        img_filename,
                    ]

                    src_img_name = None
                    for cand in image_candidates:
                        if cand in names:
                            src_img_name = cand
                            break

                    if src_img_name is None:
                        # fallback: brute-force basename match
                        for n in names:
                            if n.lower().endswith((".jpg", ".jpeg", ".png")) and Path(n).name == img_filename:
                                src_img_name = n
                                break

                    if src_img_name is None:
                        print(f"[WARNING] Image not found for XML: {xml_name}")
                        continue

                    # Flatten output filename to avoid collisions across weather folders
                    dst_stem = f"{weather}_{Path(img_filename).stem}"
                    dst_img_name = f"{dst_stem}{Path(img_filename).suffix.lower()}"
                    dst_img_path = out_img_dir / dst_img_name

                    if not dst_img_path.exists():
                        with open(dst_img_path, "wb") as f:
                            f.write(inner_zip.read(src_img_name))
                        stats["images_written"] += 1

                    label_lines = []

                    for obj in parsed["objects"]:
                        cls_name = obj["name"]

                        if cls_name not in CLASS_MAP:
                            stats["unknown_classes"][cls_name] += 1
                            continue

                        cls_id = CLASS_MAP[cls_name]

                        xc, yc, bw, bh = voc_to_yolo(
                            obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                            parsed["width"], parsed["height"]
                        )

                        xc = clamp(xc)
                        yc = clamp(yc)
                        bw = clamp(bw)
                        bh = clamp(bh)

                        if bw <= 0 or bh <= 0:
                            continue

                        label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                        stats["objects_kept"] += 1
                        stats["class_counts"][cls_id] += 1
                        stats["weather_counts"][weather] += 1

                    label_path = out_lbl_dir / f"{dst_stem}.txt"
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(label_lines))
                    stats["labels_written"] += 1

    write_yaml()

    print("\n================ FINAL SUMMARY ================")
    print(f"Images written : {stats['images_written']}")
    print(f"Labels written : {stats['labels_written']}")
    print(f"Objects kept   : {stats['objects_kept']}")
    print(f"XML files seen : {stats['xml_files_seen']}")

    print("\nClass counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name:<12} {stats['class_counts'][i]}")

    print("\nWeather counts:")
    for weather in ["fog", "rain", "snow", "sand"]:
        print(f"  {weather:<6}: {stats['weather_counts'][weather]}")

    if stats["unknown_classes"]:
        print("\nUnknown / ignored classes:")
        for k, v in sorted(stats["unknown_classes"].items()):
            print(f"  {k:<15} {v}")

    print(f"\nYOLO DAWN 8-class dataset ready at:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    convert()
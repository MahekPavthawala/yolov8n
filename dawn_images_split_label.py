# -*- coding: utf-8 -*-
"""
Convert DAWN original annotations to YOLO 8-class format
and split directly into train / val / test.

This script:
1) reads DAWN from the original outer zip
2) reads Pascal VOC XML annotations from each weather zip
3) maps DAWN classes to the 8 YOLO classes
4) splits images into train/val/test
5) writes images + labels directly into YOLO folders
6) writes dawn_8cls.yaml

Author: mahek
"""

from pathlib import Path
import zipfile
import io
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================================================
# UPDATE PATHS
# ============================================================
DAWN_ZIP = Path(r"C:\UDMercy\Semester 1\Project\766ygrbt8y-3.zip")
OUTPUT_ROOT = Path(r"C:\UDMercy\Semester 1\Project\dawn_yolo_8cls")
# ============================================================

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

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

# DAWN label -> your 8-class mapping
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
    # rider usually absent in DAWN
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


def prepare_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml():
    yaml_path = OUTPUT_ROOT / "dawn_8cls.yaml"
    yaml_text = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
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
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\nYAML written to: {yaml_path}")


def collect_all_records():
    records = []
    unknown_classes = defaultdict(int)

    with zipfile.ZipFile(DAWN_ZIP, "r") as outer_zip:
        for weather, inner_zip_name in INNER_ZIPS.items():
            if inner_zip_name not in outer_zip.namelist():
                print(f"[WARNING] Missing inner zip: {inner_zip_name}")
                continue

            print(f"\nReading {weather}...")

            inner_bytes = outer_zip.read(inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as inner_zip:
                names = inner_zip.namelist()
                xml_files = [n for n in names if "PASCAL_VOC/" in n and n.lower().endswith(".xml")]

                for xml_name in xml_files:
                    parsed = parse_voc_xml(inner_zip.read(xml_name))
                    if parsed is None or not parsed["filename"]:
                        continue

                    img_filename = parsed["filename"]

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
                        for n in names:
                            if n.lower().endswith((".jpg", ".jpeg", ".png")) and Path(n).name == img_filename:
                                src_img_name = n
                                break

                    if src_img_name is None:
                        continue

                    label_lines = []
                    class_ids_present = []

                    for obj in parsed["objects"]:
                        cls_name = obj["name"]

                        if cls_name not in CLASS_MAP:
                            unknown_classes[cls_name] += 1
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
                        class_ids_present.append(cls_id)

                    dst_stem = f"{weather}_{Path(img_filename).stem}"
                    dst_img_name = f"{dst_stem}{Path(img_filename).suffix.lower()}"

                    records.append({
                        "weather": weather,
                        "dst_stem": dst_stem,
                        "dst_img_name": dst_img_name,
                        "src_img_name": src_img_name,
                        "label_lines": label_lines,
                        "class_ids_present": class_ids_present,
                        "image_bytes_zip_name": src_img_name,
                        "inner_zip_name": inner_zip_name,
                    })

    return records, unknown_classes


def split_records(records):
    random.seed(SEED)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def write_split(split_records):
    stats = {
        "images_written": 0,
        "labels_written": 0,
        "objects_kept": 0,
        "class_counts": [0] * len(CLASS_NAMES),
        "split_counts": {},
    }

    prepare_dirs()

    with zipfile.ZipFile(DAWN_ZIP, "r") as outer_zip:
        inner_zip_cache = {}

        for split, records in split_records.items():
            stats["split_counts"][split] = len(records)

            out_img_dir = OUTPUT_ROOT / "images" / split
            out_lbl_dir = OUTPUT_ROOT / "labels" / split

            for rec in records:
                inner_zip_name = rec["inner_zip_name"]

                if inner_zip_name not in inner_zip_cache:
                    inner_bytes = outer_zip.read(inner_zip_name)
                    inner_zip_cache[inner_zip_name] = zipfile.ZipFile(io.BytesIO(inner_bytes), "r")

                inner_zip = inner_zip_cache[inner_zip_name]

                dst_img_path = out_img_dir / rec["dst_img_name"]
                dst_lbl_path = out_lbl_dir / f"{rec['dst_stem']}.txt"

                if not dst_img_path.exists():
                    with open(dst_img_path, "wb") as f:
                        f.write(inner_zip.read(rec["src_img_name"]))
                    stats["images_written"] += 1

                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(rec["label_lines"]))
                stats["labels_written"] += 1

                stats["objects_kept"] += len(rec["label_lines"])
                for cls_id in rec["class_ids_present"]:
                    stats["class_counts"][cls_id] += 1

        for z in inner_zip_cache.values():
            z.close()

    return stats


def main():
    print("Collecting DAWN annotations...")
    records, unknown_classes = collect_all_records()

    if len(records) == 0:
        raise RuntimeError("No DAWN records were collected. Check zip paths.")

    print(f"\nTotal usable DAWN records: {len(records)}")

    splits = split_records(records)
    stats = write_split(splits)
    write_yaml()

    print("\n================ DAWN CONVERSION + SPLIT SUMMARY ================\n")
    print(f"Images written : {stats['images_written']}")
    print(f"Labels written : {stats['labels_written']}")
    print(f"Objects kept   : {stats['objects_kept']}")

    print("\nSplit counts:")
    for split, count in stats["split_counts"].items():
        print(f"  {split:<5}: {count}")

    print("\nClass counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name:<12} {stats['class_counts'][i]}")

    if unknown_classes:
        print("\nUnknown / ignored classes:")
        for k, v in sorted(unknown_classes.items()):
            print(f"  {k:<15} {v}")

    print(f"\nOutput root:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
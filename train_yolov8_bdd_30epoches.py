# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:23:31 2026

@author: mahek
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics import YOLO

DATA_YAML = r"C:\UDMercy\Semester 1\Project\bdd100k_yolo_3cls\bdd100k.yaml"

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,          # you can increase from 0 now
        pretrained=True,
        project=r"C:\UDMercy\Semester 1\Project\runs_bdd",
        name="yolov8n_bdd3cls_30ep",
        patience=10,
        optimizer="auto",
        close_mosaic=10
    )

if __name__ == "__main__":
    main()
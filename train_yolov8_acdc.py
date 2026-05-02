# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 22:04:43 2026

@author: mahek
"""

from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\UDMercy\Semester 1\Project\ACDC\yolo_acdc_fog\data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,   # safest in Spyder/Windows
        name="acdc_fog_exp1"
    )


if __name__ == "__main__":
    freeze_support()
    main()
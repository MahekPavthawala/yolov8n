# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:37:47 2026

@author: mahek
"""

from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\UDMercy\Semester 1\Project\ACDC\yolo_acdc_all_weather\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name="acdc_all_weather_yolov8n_50e"
    )


if __name__ == "__main__":
    freeze_support()
    main()
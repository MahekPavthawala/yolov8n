# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:02:01 2026

@author: mahek
"""

from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO(
        r"C:\Workplace Documents\PhD\UDMercy\Semester 1\ELEE5940 Advance Deep Learning for Computer Vision\Course Project\code\runs\detect\acdc_all_weather_yolov8s_50e\weights\best.pt"
    )

    metrics = model.val(
        data=r"C:\UDMercy\Semester 1\Project\ACDC\yolo_acdc_all_weather\data.yaml",
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )

    print(metrics.results_dict)
    print(metrics.maps)


if __name__ == "__main__":
    freeze_support()
    main()
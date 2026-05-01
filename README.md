# Robust Object Detection and Lane Segmentation under Adverse Weather

This project investigates the robustness of YOLOv8-based perception systems for autonomous driving under adverse weather conditions such as fog, rain, snow, and nighttime scenarios.

The study focuses on two critical perception tasks:
- Object Detection
- Lane Segmentation

It evaluates how performance degrades under domain shift and explores domain adaptation techniques to improve robustness.

---

## 🚀 Key Features

- YOLOv8n object detection pipeline
- YOLOv8n-seg lane segmentation pipeline
- Cross-dataset evaluation (BDD100K → ACDC → DAWN)
- Domain adaptation via fine-tuning
- Reverse generalization analysis
- Pseudo-labeling for lane segmentation
- Failure analysis under adverse weather conditions

---

## 📊 Datasets

This project uses the following datasets:

- **BDD100K** – Clear-weather dataset used for training
- **ACDC** – Adverse weather dataset (fog, rain, snow, night)
- **DAWN** – Cross-dataset evaluation benchmark

> ⚠️ Note: Datasets are not included in this repository due to size. Please download them from their official sources.

---

## 🧱 Project Structure

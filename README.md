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

> ⚠️ Note: Datasets are not included in this repository due to size.

---

## 🧱 Project Structure

```text
yolov8n/
├── Code/
├── Documents/
├── Sample Outputs/
├── README.md
├── requirements.txt
└── LICENSE

```

## ⚙️ Installation

```bash
pip install ultralytics
pip install opencv-python numpy pandas matplotlib
---
```

## 🧪 Usage

Train Object Detection:
python train_detection.py

Evaluate Model:
python evaluate.py

Generate Pseudo Labels:
python generate_pseudo_labels.py

Train Lane Segmentation:
python train_segmentation.py

---

## 📈 Results

Object Detection:

- Domain shift causes performance degradation
- Fine-tuning improves robustness
- Cross-dataset generalization is limited
- Asymmetric generalization observed

Lane Segmentation:

- Pseudo-labeling enables adaptation
- Improved segmentation continuity
- Snow shows highest failure rate

---

## 🧪 Usage

### Dataset Preparation

```bash
python Code/convert_BDD100K_coco_to_yolo_multiclass.py
python Code/convert_ACDC_coco_to_yolo_multiclass.py
python Code/dawn_images_split_label.py
```

### Object Detection

```bash
python Code/YOLOv8n_train_BDD100K_test_ACDC.py
python Code/finetune_yolov8n_bdd_to_acdc_final.py
python Code/yolov8n_train_validate_acdc_test_dawn.py
python Code/yolov8n_train_validate_dawn_test_acdc.py
```

### Lane Segmentation

```bash
python Code/train_yolov8n_bdd_lane_segmentation.py
python Code/generate_acdc_pseudo_labels.py
python Code/fine_tune_yolov8n_pseudo_label_acdc.py
python Code/infer_ACDC_lane__weatherwise_yolov8seg.py
```
---
## 📊 Datasets

Datasets are not included in this repository due to size.

BDD100K
https://bdd-data.berkeley.edu/

ACDC
https://acdc.vision.ee.ethz.ch/

DAWN
https://data.mendeley.com/datasets/766ygrbt8y/3

## 🔥 Key Findings

Domain shift impacts perception
Fine-tuning helps but not fully
Generalization is asymmetric
Failure scenarios always exist

---

## 🔮 Future Work

Multi-dataset training
Domain generalization
Transformer-based models
Real-time optimization

---

## 👨‍💻 Author

Mahekkumar Pavthawala  
University of Detroit Mercy

---

## 📜 License

MIT License
****

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:00:32 2026

@author: mahek
"""

import matplotlib.pyplot as plt

metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
acdc = [0.478, 0.317, 0.323, 0.194]
dawn = [0.330, 0.159, 0.147, 0.0835]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], acdc, width=width, label="ACDC val")
plt.bar([i + width/2 for i in x], dawn, width=width, label="DAWN test")

plt.xticks(list(x), metrics)
plt.ylabel("Score")
plt.title("Cross-dataset robustness: ACDC-trained YOLOv8n")
plt.legend()
plt.tight_layout()
plt.savefig("acdc_vs_dawn_metric_drop.png", dpi=300, bbox_inches="tight")
plt.show()
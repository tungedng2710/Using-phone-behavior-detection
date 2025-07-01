# Model Zoo

This document provides a summary of the trained models for phone usage behavior detection.

## Models

| Model | Dataset | Epochs | Image Size | mAP50-95 | mAP50 | Precision | Recall | Weights |
|---|---|---|---|---|---|---|---|---|
| YOLOv11l | mobile_phone_v1.2 | 20 | 640 | 0.730 | 0.964 | 0.940 | 0.915 | [yolo11l.pt](runs/detect/yolo11l_mobile_phone_v1.2_20eps_640_2025-06-30_16-45-05/weights/best.pt) |
| YOLOv11m | mobile_phone_v1.2 | 30 | 640 | 0.769 | 0.976 | 0.963 | 0.937 | [yolo11m.pt](runs/detect/yolo11m_mobile_phone_v1.2_30eps_640_2025-07-01_10-23-59/weights/best.pt) |

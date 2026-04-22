# CPPHM - Fruit Detection and Classification System

This repository contains source scripts for a fruit detection and classification system using YOLOv8.

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Weights
The model weights (`.pt` files) are not included in this repository due to size limits. You should:
- Download the base models (e.g., `yolov8n.pt`, `yolov8n-cls.pt`) automatically by running the training/inference scripts.
- If you have custom trained weights, place them in the root directory.

### 4. Datasets
Datasets (like `fruits-360`) are excluded from this repository. 
- You can use the `scrape_fruits.py` script to collect new data.
- Or use the `prepare_*.py` scripts to organize your local dataset into the YOLO format.

### 5. Running the System
- **Webcam Detection:** Run `webcam_detect.py` or `webcam_od_25.py` for real-time detection.
- **Training:** Run `train_demo.py` or `train_od_25.py` to start training on your local dataset.

## Project Structure
- `prepare_*.py`: Scripts for data preprocessing and splitting.
- `train_*.py`: Training scripts for different models.
- `webcam_*.py`: Inference scripts for real-time detection via webcam.
- `scrape_fruits.py`: Tool for scraping images for new classes.

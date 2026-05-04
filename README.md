# YOLO vs Faster R-CNN for F1TENTH Vehicle Detection

This repository contains the code and results for a 24-788 Introduction to Deep Learning mini-project comparing two object detection methods on the F1TENTH vehicle detection dataset.

The project compares:

1. **YOLO baseline**: a lightweight one-stage detector adapted from the F1TENTH Lab 8 vision notebook. (https://github.com/f1tenth-cmu/f1tenth_lab8/tree/main)
2. **Faster R-CNN variant**: a two-stage detector using a MobileNetV3-Large Feature Pyramid Network backbone.

The goal is to compare the speed and detection quality tradeoff between a fast one-stage detector and a more accurate two-stage detector for F1TENTH vehicle perception.

---

## Repository Structure

```text
deep_learning_yolo_vs_faster_RCNN/
│
├── checkpoints/
│   ├── fasterrcnn_best.pt
│   └── fasterrcnn_last.pt
|
├── data/
│   ├── f110_dataset_20220209/
│
├── notebooks/
│   ├── f110_yolo_baseline.ipynb
│   ├── fasterrcnn_f110_project.ipynb
│   ├── yolo_v_fastrcnn.ipynb
│   ├── model_100.pt
│   ├── model_300.pt
│   ├── loss_100.npy
│   └── loss_300.npy
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── splits/
│   ├── train_indices.npy
│   └── val_indices.npy
│
├── src/
│
├── .gitignore
└── README.md
```

---

## How to reproduce key results

### Environment Setup
Use a conda environment for training:
```bash
conda create -n f110_detection python=3.10 -y
conda activate f110_detection
pip install numpy pandas matplotlib opencv-python pillow tqdm jupyter notebook ipykernel
pip install torch torchvision
```
You can check out the reproduce_results.ipynb file in notebooks to load the models and look at the result without training the models. This file should run the current models and produce the tables and figures.
---
### Get the data
If you want to train the model yourself and try to replicate this project you will need to first download the data from the following github (The data is large so it is excluded from the repo):
```text
https://github.com/f1tenth-cmu/f1tenth_lab8/tree/main
```
Then place the data under the data folder as seen in the respository structure described above. The following file location should be accessible:

```text
data/f110_dataset_20220209/labels.npy
```
### Split the data
You can use the src/split_data.py script to set your rnd seed and the training-validation ratio. This will result in the data split files located here:
```text
splits/train_indicies.npy
splits/val_indicies.npy
```

### Get the models

After getting all the dependencies and the split data you can run the following files to get the models:
```text
notebooks/f110_yolo_baseline.ipynb
notebooks/fasterrcnn_f110_project.ipynb
```
This should give you the following YOLO model files:
```text
notebooks/loss_[# of epochs].npy
notebooks/model_[# of epochs].npy
```
and the following files for the faster R-CNN:
```text
checkpoints/fasterrcnn_best.pt
checkpoints/fasterrcnn_last.pt
```
After aquiring this data, you may use the reproduce_results.ipynb file to get the results
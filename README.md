# Artist Classification with PyTorch

This project builds a deep learning model that predicts the artist of a painting using a fine‑tuned **ResNet‑50** model in PyTorch.

---

## 📌 Overview

This repository contains:

* Training pipeline using transfer learning (ResNet‑50)
* Safe dataset loader that skips corrupted images
* Image pre‑processing & augmentation
* Model saving with class labels
* Inference script for predicting artist from new paintings

The model is trained on a dataset of paintings and learns to classify each image into its corresponding artist.

---

## 🧠 Model Architecture

* Backbone: **ResNet‑50** pretrained on ImageNet
* Final FC layer replaced for `num_classes` artists
* Backbone frozen (feature extractor mode)
* Optimizer: Adam (`lr = 0.001`)
* Loss: Cross‑Entropy

---

## 📂 Dataset

The dataset is **not included** in this repository due to size. It will be provided via Google Drive link.

Once downloaded, place it like this:

```
dataset/
 ├── train/
 │    ├── artist_1/
 │    ├── artist_2/
 │    └── ...
 └── val/
      ├── artist_1/
      ├── artist_2/
      └── ...
```

dataset/
├── train/
│    ├── artist_1/
│    ├── artist_2/
│    └── ...
└── val/
├── artist_1/
├── artist_2/
└── ...

```

---
### 🔧 Model Weights (.pth)
The trained `.pth` model file is also **not included** in this repository due to size. It will be provided via a Google Drive link.

Once downloaded, place it in the project root and ensure your inference script points to it:
```

model_path = "artist_model.pth"

```

## 🚀 Training
Run the training script to train the artist classifier.

The script automatically:
- Detects device (MPS / CUDA / CPU)
- Loads dataset
- Applies augmentations
- Trains and evaluates model
- Saves best model as `artist_model.pth`

---
## 🧪 Inference
Place your test image (e.g., `test.jpg`) and run the inference section to get:
- Predicted artist ✅
- Confidence score ✅
- Top‑3 predictions ✅

---
## ✅ Features
- Handles broken images gracefully
- Reproducible workflows
- Lightweight transfer learning
- Apple Silicon (MPS) support
- Easy prediction utility

---
## 📎 Requirements
- Python 3.8+
- PyTorch
- torchvision
- PIL
- OS supporting GPU acceleration (optional)

Install required packages:
```

pip install torch torchvision pillow

```

---
## 🏁 Output Example
```

🎨 Predicted artist: Vincent van Gogh
🔒 Confidence: 0.87
🔎 Top 3 guesses:

* Vincent van Gogh (0.874)
* Claude Monet (0.054)
* Paul Cézanne (0.032)

```

---
## 🤝 Contributing
Pull requests and improvements are welcome!

---
## 📜 License
MIT License

---
## ✨ Future Work
- Unfreeze backbone for fine‑tuning
- Add style‑transfer augmentation
- Add Gradio web UI for demo
- Convert to ONNX / CoreML

---
## 🏷️ Suggested Academic Model Name
> **ArtAttributionNet (AANet)** — Artist Attribution Neural Network

```

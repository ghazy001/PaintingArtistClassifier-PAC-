
# 🎨 Artist Attribution with PyTorch (ResNet-50)

This project uses **transfer learning** with **ResNet-50** to classify paintings by artist.  
A deep learning model is trained on a dataset of paintings and predicts the most likely creator.

---

## 📌 Features

- ✅ Transfer Learning (ResNet-50 pretrained on ImageNet)
- ✅ Safe image loader (skips corrupted images)
- ✅ Apple Silicon (M-series), CUDA & CPU support
- ✅ Top-3 prediction output
- ✅ Training + Inference scripts provided
- ✅ Colab notebook for cloud training

---

## 📂 Project Structure

```

📦 artist-classification
┣ 📁 dataset/        # dataset folder (not included)
┣ 📄 train.py        # training script
┣ 📄 predict.py    # prediction script
┣ 📄 artist_model.pth (not included)
┣ 📄 README.md
┣ 📄 test.jpg

```

---

## 📥 Dataset

The dataset is **not included** in this repository due to size restrictions.  
It will be shared via **Drive link**.

> 📎 **Dataset Download:** [_data set link_](https://drive.proton.me/urls/Q41Z6YNWCW#GOGRJRocpI3B)

Place it like this after downloading:

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

## 🧠 Model Weights (.pth)

The trained model weights are not included here.

> 📦 **Model Weights Download:** _Google Drive link coming soon_

Place the file in the project root:

```

artist_model.pth

````

---

## 🚀 Training

Run training locally:

```bash
python3 train.py
````

The script will:

* Detect GPU / MPS / CPU
* Load images
* Train ResNet-50
* Save `artist_model.pth`

---

## 🔍 Inference

Run classification on a test image:

```bash
python3 predict.py 
```

Expected output:

```
🎨 Predicted artist: Vincent van Gogh
🔒 Confidence: 0.87
🔎 Top 3 guesses:
 - Vincent van Gogh (0.874)
 - Claude Monet (0.054)
 - Paul Cézanne (0.032)
```


---

## 📦 Requirements

Install dependencies:

```bash
pip install torch torchvision pillow
```



---

## 📜 License

MIT License — feel free to use, modify, and contribute!


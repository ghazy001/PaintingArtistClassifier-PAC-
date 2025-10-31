import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model_path = "artist_model.pth"

# Choose device (MPS for Apple GPU, then CUDA, then CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Preprocessing must match validation transforms from training
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(checkpoint_path):
    # 1. load the checkpoint dict
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 2. rebuild the same model architecture
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 3. load only the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, class_names

def predict_image(img_path, model, class_names):
    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(tensor)  # shape [1, num_classes]
        probs = torch.softmax(outputs, dim=1)[0]  # shape [num_classes]
        top_prob, top_idx = torch.max(probs, dim=0)

    predicted_artist = class_names[top_idx.item()]
    confidence = float(top_prob.item())

    # Optional: also get top 3
    top3_prob, top3_idx = torch.topk(probs, k=3)
    top3 = []
    for p, idx in zip(top3_prob, top3_idx):
        top3.append({
            "artist": class_names[idx.item()],
            "confidence": float(p.item())
        })

    return predicted_artist, confidence, top3

if __name__ == "__main__":
    # Load model + classes from checkpoint
    model, class_names = load_model(model_path)

    # Put your test image path here
    test_image_path = "test.jpg"

    artist, conf, top3 = predict_image(test_image_path, model, class_names)

    print("🎨 Predicted artist:", artist)
    print("🔒 Confidence:", conf)
    print("🔎 Top 3 guesses:")
    for guess in top3:
        print(f"  - {guess['artist']} ({guess['confidence']:.3f})")

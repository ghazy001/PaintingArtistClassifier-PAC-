import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import UnidentifiedImageError
import os, copy

########################################
# 1. Device selection (M1 / CUDA / CPU)
########################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

########################################
# 2. Basic config
########################################
data_dir = "/Users/ghazi.saoudi/Downloads/dataset"  # your dataset root
batch_size = 8
num_epochs = 15
learning_rate = 0.001

########################################
# 3. Dataset class that skips bad images
########################################
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError, ValueError):
            # If an image is broken, move to the next one
            new_index = (index + 1) % len(self.samples)
            return super().__getitem__(new_index)

########################################
# 4. Transforms
########################################
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

########################################
# 5. Datasets and loaders
########################################
image_datasets = {
    'train': SafeImageFolder(os.path.join(data_dir, 'train'), train_transforms),
    'val': SafeImageFolder(os.path.join(data_dir, 'val'), val_transforms)
}

dataloaders = {
    'train': DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # macOS + MPS prefers 0 workers
        pin_memory=False
    ),
    'val': DataLoader(
        image_datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
}

dataset_sizes = {
    phase: len(image_datasets[phase])
    for phase in ['train', 'val']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)
print("Detected classes:", class_names)
print("num_classes =", num_classes)

########################################
# 6. Load pretrained model and modify head
########################################
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer with our num_classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

########################################
# 7. Training loop
########################################
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0.0  # float, not tensor

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)               # [batch, num_classes]
                _, preds = torch.max(outputs, 1)      # predicted class index
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # stats
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / float(dataset_sizes[phase])

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Track best on validation set
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

print(f"Best val Acc: {best_acc:.4f}")

########################################
# 8. Save best model (with class names)
########################################
model.load_state_dict(best_model_wts)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, "artist_model.pth")

print("✅ Training finished! Model saved as artist_model.pth")

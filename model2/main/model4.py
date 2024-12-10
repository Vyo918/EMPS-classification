import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Adjust the dataset path here
DATASET_PATH = "./dataset/my_dataset/model2/FashionStyle14/data_csv_files/"

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets and DataLoaders
train_dataset = datasets.ImageFolder(root=f"{DATASET_PATH}/train", transform=transform_train)
val_dataset = datasets.ImageFolder(root=f"{DATASET_PATH}/val", transform=transform_val_test)
test_dataset = datasets.ImageFolder(root=f"{DATASET_PATH}/test", transform=transform_val_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Swin Transformer Base Model
model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
model.head = nn.Linear(model.head.in_features, len(train_dataset.classes))  # Adjust for number of classes
model = model.to(device)
print("Model's device: ", next(model.parameters()).device)  # Should print 'cuda:0'

# Criterion and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

# Training, validation, and testing functions
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        assert str(images.device) == str(device), f"Images on {images.device}, but expected {device}"
        assert str(labels.device) == str(device), f"Labels on {labels.device}, but expected {device}"
        assert next(model.parameters()).device == device, "Model not on the correct device"

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total

def evaluate_epoch(model, loader, criterion, desc="Evaluating"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    return running_loss / len(loader), correct / total, np.array(all_labels), np.array(all_preds)

# Training and validation loop
best_acc = 0.0
best_model_path = "./best_model4.pth"
epochs = 10  # Adjust based on preference

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_labels, val_preds = evaluate_epoch(model, val_loader, criterion, desc="Validating")

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {(train_acc*100):.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {(val_acc*100):.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved Best Model with Acc: {best_acc:.4f}")

# Testing the model
print("\nTesting the best model...")
model.load_state_dict(torch.load(best_model_path))
test_loss, test_acc, test_labels, test_preds = evaluate_epoch(model, test_loader, criterion, desc="Testing")
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Plot confusion matrix for test dataset
conf_matrix = confusion_matrix(test_labels, test_preds, labels=np.arange(len(test_dataset.classes)))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=test_dataset.classes)
disp.plot(cmap="viridis")
plt.show()

print(f"Best Model Saved at {best_model_path}")

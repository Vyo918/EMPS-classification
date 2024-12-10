#Response 2 from chatgpt to implement paper
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Define Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets

data_path = "./dataset/my_dataset/model2/FashionStyle14/data_csv_files/"

train_dir = data_path + "train/"
val_dir = data_path + "val/"
test_dir = data_path + "test/"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# IRSN Model Definition
class IRSN(nn.Module):
    def __init__(self, num_classes):
        super(IRSN, self).__init__()
        # Domain-specific feature extractor (Swin Transformer)
        self.domain_specific_backbone = models.swin_v2_b(weights="IMAGENET1K_V1")
        self.domain_specific_backbone.head = nn.Identity()  # Remove classification head
        
        # General feature extractor (CLIP Vision Encoder or equivalent)
        self.general_backbone = models.swin_v2_b(weights="IMAGENET1K_V1")  # Replace with CLIP encoder if available
        self.general_backbone.head = nn.Identity()  # Remove classification head

        # Item-specific encoders (head, top, bottom, shoes)
        self.item_encoders = nn.ModuleDict({
            "head": nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256)),
            "top": nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256)),
            "bottom": nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256)),
            "shoes": nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256)),
        })

        # Gated Feature Fusion (GFF)
        self.gff = nn.Sequential(
            nn.Linear(512 * 4 + 1024 + 512, 512),  # Combine item and global features
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, item_masks):
        """
        Args:
            x: Input image batch
            item_masks: A dict of masks for head, top, bottom, shoes
        """
        # Global features
        global_features = self.domain_specific_backbone(x)
        general_features = self.general_backbone(x)

        # Item features
        item_features = []
        for region, encoder in self.item_encoders.items():
            mask = item_masks[region]
            masked_feature = global_features * mask.unsqueeze(1)  # Apply mask
            pooled_feature = nn.AdaptiveAvgPool2d((1, 1))(masked_feature).view(x.size(0), -1)
            encoded_feature = encoder(pooled_feature)
            item_features.append(encoded_feature)

        # Fuse all features
        all_features = torch.cat(item_features + [global_features, general_features], dim=1)
        output = self.gff(all_features)
        return output

# Generate Item Masks Placeholder
# Replace with an actual segmentation model like CLIPSeg
def generate_item_masks(batch):
    """
    Generate item-specific binary masks for head, top, bottom, and shoes.
    This is a placeholder; replace with an actual segmentation model.
    """
    masks = {
        "head": torch.ones(batch.size(0), 1, 224, 224).to(batch.device),
        "top": torch.ones(batch.size(0), 1, 224, 224).to(batch.device),
        "bottom": torch.ones(batch.size(0), 1, 224, 224).to(batch.device),
        "shoes": torch.ones(batch.size(0), 1, 224, 224).to(batch.device),
    }
    return masks

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IRSN(num_classes=len(train_dataset.classes)).to(device)

# Optimizer, Loss, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            item_masks = generate_item_masks(inputs)

            optimizer.zero_grad()
            outputs = model(inputs, item_masks)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                item_masks = generate_item_masks(inputs)
                outputs = model(inputs, item_masks)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)

        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        scheduler.step()

    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)
    return model

# Train the Model
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

# Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            item_masks = generate_item_masks(inputs)
            outputs = model(inputs, item_masks)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

labels, preds = evaluate_model(model, test_loader)
acc = accuracy_score(labels, preds)
print(f"Test Accuracy: {acc:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()

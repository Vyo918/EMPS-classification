import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny
from utils import train, evaluate, conf_mat, plot_learning_curves

def train_all_model(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, GRAPH_PATH, TEXT_PATH):
    # Model Components
    class CrissCrossAttention(nn.Module):
        def __init__(self, in_channels):
            super(CrissCrossAttention, self).__init__()
            self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            batch_size, channels, height, width = x.size()
            query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
            key = self.key_conv(x).view(batch_size, -1, height * width)
            value = self.value_conv(x).view(batch_size, -1, height * width)

            attention = torch.bmm(query, key)
            attention = self.softmax(attention)

            out = torch.bmm(value, attention.permute(0, 2, 1))
            out = out.view(batch_size, channels, height, width)

            return x + out

    class SpatialAttention(nn.Module):
        def __init__(self, in_channels):
            super(SpatialAttention, self).__init__()
            self.conv = nn.Conv2d(2, in_channels, kernel_size=7, padding=3)  # Match in_channels
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv(x)
            return x * self.sigmoid(x)

    class ConvNeXtWithAttention(nn.Module):
        def __init__(self, num_classes):
            super(ConvNeXtWithAttention, self).__init__()
            self.convnext = convnext_tiny(pretrained=True)
            self.convnext.features[0][0] = nn.Conv2d(3, 96, kernel_size=4, stride=4, padding=0)

            self.criss_cross_attention = CrissCrossAttention(768)
            self.spatial_attention = SpatialAttention(in_channels=768)

            # Fully connected layers with correct input size
            self.fc = nn.Sequential(
                nn.Linear(768, 512),  # Ensure input matches global pooling output
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.convnext.features(x)  # Extract features
            x = self.criss_cross_attention(x)  # Apply Criss Cross Attention
            x = self.spatial_attention(x)  # Apply Spatial Attention
            x = torch.mean(x, dim=[2, 3])  # Global Average Pooling to reduce to (batch_size, 768)
            x = self.fc(x)  # Pass through fully connected layers
            return x
            
    print(f"Using device: {DEVICE}")

    # data transforms for train and test datasets
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dir = DATA_PATH + "train"
    trainset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    
    val_dir = DATA_PATH + "val"
    valset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    val_dataloader = DataLoader(valset, batch_size=32, shuffle=False)
    
    test_dir = DATA_PATH + "test"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)

    model = ConvNeXtWithAttention(num_classes=len(testset.classes)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    
    best_acc = 0
    epochs = 25
    patience = 5  # Early stopping patience
    no_improve = 0  # Counter for early stopping
    history = {"train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
            }
    
    with open(TEXT_PATH, 'w') as f:
        for epoch in range(epochs):
            f.write(f'Epoch {epoch + 1}/{epochs}:\n')
            print(f'Epoch {epoch + 1}/{epochs}:')
            
            train_loss, train_acc = train(model=model,
                                        data_loader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=DEVICE)
            f.write(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%\n')
            print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
            
            val_loss, val_acc, labels, preds = evaluate(model=model,
                                                        data_loader=val_dataloader,
                                                        loss_fn=loss_fn,
                                                        device=DEVICE)
            
            f.write(f'Testing Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n')
            print(f'Testing Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.jit.script(model).save(MODEL_PATH)
                flag = False
                
                f.write(f'Testing Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n')
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
                no_improve = 0  # Reset early stopping counter
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered!")
                    break
            scheduler.step()
            f.write('\n')
            print()
        
    final_acc, labels, preds = conf_mat(model=model,
                                        data_loader=test_dataloader,
                                        dataset=testset,
                                        device=DEVICE,
                                        model_path=MODEL_PATH,
                                        confmat_path=CONFMAT_PATH)
    plot_learning_curves(history, final_acc, graph_path=GRAPH_PATH)   
    return final_acc

def main():
    print(f"Training top model:\n")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = f"./dataset/my_dataset/model2/split/"
    MODEL_PATH = f"./classification_model/model2/model/train.pt"
    CONFMAT_PATH = f"./classification_model/model2/result/confmat/train.jpg"
    GRAPH_PATH = f"./classification_model/model2/result/graph/train.jpg"
    TEXT_PATH = f"./classification_model/model2/result/text/train.txt"
    acc = train_all_model(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, GRAPH_PATH, TEXT_PATH)
    print("-"*50)
    print(f"Accuracy of trained model: {acc:.2f}%")
if __name__ == "__main__":
    main()
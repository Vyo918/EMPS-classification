import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from pathlib import Path
from rich.progress import track

device = "cuda" if torch.cuda.is_available() else "cpu"

# Criss Cross Attention Module
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.conv_q(x)  # Query
        k = self.conv_k(x)  # Keydc
        v = self.conv_v(x)  # Value

        q_flatten = q.view(B, -1, H * W)
        k_flatten = k.view(B, -1, H * W)
        attention = torch.bmm(q_flatten.permute(0, 2, 1), k_flatten)  # Criss-Cross Attention
        attention = self.softmax(attention)
        v_flatten = v.view(B, -1, H * W)
        out = torch.bmm(v_flatten, attention)
        out = out.view(B, C, H, W)

        return out + x  # Residual connection

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return out * x

# ConvNeXt Block
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNeXtBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)  # Adjust layer normalization size accordingly
        self.pointwise_conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        self.gelu = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        return x + residual  # Residual connection

# the complete ConvNeXt model
class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtTiny, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.block1 = ConvNeXtBlock(96, 96)
        self.attn1 = CrissCrossAttention(96)  # first attention module

        self.downsample1 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
        self.block2 = ConvNeXtBlock(192, 192)

        self.downsample2 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
        self.block3 = ConvNeXtBlock(384, 384)

        self.downsample3 = nn.Conv2d(384, 768, kernel_size=2, stride=2)
        self.block4 = ConvNeXtBlock(768, 768)
        self.attn2 = SpatialAttention(768)  # second attention module

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.attn1(x)

        x = self.downsample1(x)
        x = self.block2(x)

        x = self.downsample2(x)
        x = self.block3(x)

        x = self.downsample3(x)
        x = self.block4(x)
        x = self.attn2(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# training
def train(model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = device
        ):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in track(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        output = model(images)
        loss = loss_fn(output, labels)
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
# Testing
def evaluate(model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device = device
        ):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in track(data_loader):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

# data transforms for train and test datasets
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

data_path = "./dataset/my_dataset/model2/FashionStyle14/data_csv_files/"

train_dir = data_path + "train/"
test_dir = data_path + "test/"

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes

model = ConvNeXtTiny(num_classes=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
best_acc = 0.0    
epochs = 50
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 100)
    train(model, train_dataloader, loss_fn, optimizer, device)
    acc = evaluate(model, test_dataloader, loss_fn, device)
    if acc > best_acc:
        best_acc = acc
        # torch.save(model.state_dict(), f"best_main.pth")
        print(f"New best model saved with accuracy: {best_acc:.2f}%")
    print()

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
y_preds = []
model.eval()
with torch.inference_mode():
    for x, y in track(test_dataloader, desc="Making Predictions"):
        x, y = x.to(device), y.to(device)
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        y_preds.append(predicted.cpu())
y_preds = torch.cat(y_preds)    

confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_preds, target=torch.tensor(test_dataset.targets))

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10,7)
)
# plt.savefig('C:/Users/elvio/Documents/folder_vio/Coding/College/UNDERGRADUATE-PROJECT/classification_model/model2/result/main.png')
plt.show()
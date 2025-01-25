import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny
from utils import train, evaluate, conf_mat, plot_learning_curves

def train_all_model(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, GRAPH_PATH, TEXT_PATH):
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
            self.dropout1 = nn.Dropout(0.3)  # Dropout after first attention block

            self.downsample1 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
            self.block2 = ConvNeXtBlock(192, 192)

            self.downsample2 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
            self.block3 = ConvNeXtBlock(384, 384)

            self.downsample3 = nn.Conv2d(384, 768, kernel_size=2, stride=2)
            self.block4 = ConvNeXtBlock(768, 768)
            self.attn2 = SpatialAttention(768)  # second attention module
            self.dropout2 = nn.Dropout(0.3)  # Dropout after second attention block

            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(768, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.block1(x)
            x = self.attn1(x)
            x = self.dropout1(x)

            x = self.downsample1(x)
            x = self.block2(x)

            x = self.downsample2(x)
            x = self.block3(x)

            x = self.downsample3(x)
            x = self.block4(x)
            x = self.attn2(x)
            x = self.dropout2(x)

            x = self.global_avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x
            
    print(f"Using device: {DEVICE}")

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
    
    train_dir = DATA_PATH + "train"
    trainset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    
    val_dir = DATA_PATH + "val"
    valset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    val_dataloader = DataLoader(valset, batch_size=32, shuffle=False)
    
    test_dir = DATA_PATH + "test"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)

    model = ConvNeXtTiny(num_classes=len(testset.classes)).to(DEVICE)
    class_weights = torch.tensor([1.0] * len(trainset.classes)).to(DEVICE)  # Adjust weights if needed
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0
    epochs = 35
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
    print(f"Training model:\n")
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
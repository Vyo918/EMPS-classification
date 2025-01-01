import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def train(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device = device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader):
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
    
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device = device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
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

def main():
    print(f"Using device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.bn1 = nn.BatchNorm2d(6)
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            
            self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
            self.bn3 = nn.BatchNorm2d(120)
            
            self.fc1 = nn.Linear(120, 84)
            self.fc2 = nn.Linear(84, 13)
            self.dropout = nn.Dropout(p=0.5)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            x = torch.relu(self.bn3(self.conv3(x)))
            
            x = x.view(-1, 120)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset_path = "./dataset/my_dataset/model1/"
    trainset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=train_transform)
    testset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=test_transform)

    train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.apply(init_weights)
    
    best_accuracy = 92.34  # Initialize best accuracy
    epochs = 20
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        
        # Train
        train(model, train_dataloader, loss_fn, optimizer, device)
        
        # Evaluate
        accuracy = evaluate(model, test_dataloader, loss_fn, device)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"best_model.pth")
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
        
        # Step the scheduler
        scheduler.step()
        print()
    
    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix
    import matplotlib.pyplot as plt
    y_preds = []
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in tqdm(test_dataloader, desc="Making Confusion Matrix"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            y_preds.append(predicted.cpu())
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    accuracy = 100*correct/total
    print(f"Accuracy: {accuracy:.2f}%")
    
    y_preds = torch.cat(y_preds)    

    confmat = ConfusionMatrix(num_classes=len(trainset.classes), task='multiclass')
    confmat_tensor = confmat(preds=y_preds, target=torch.tensor(testset.targets))

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=trainset.classes,
        figsize=(10,7)
    )
    # plt.savefig("./classification_model/model1/result/best_accuracy.png")
    plt.show()

if __name__ == "__main__":
    main()
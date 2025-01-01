import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functions import train, evaluate, conf_mat

# device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
DATA_PATH = "./dataset/my_dataset/model1/"
MODEL_PATH = "./classification_model/model1/best_model.pth"

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class LeNet5(nn.Module):
        def __init__(self, num_classes):
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
            self.fc2 = nn.Linear(84, num_classes)
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

def main():
    print(f"Using device: {DEVICE}")

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
    train_dir = DATA_PATH + "train"
    trainset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    
    test_dir = DATA_PATH + "test"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)

    model = LeNet5(num_classes=len(testset.classes)).to(DEVICE)
    # model.load_state_dict(torch.load(f=MODEL_PATH))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model.apply(init_weights)
    
    best_acc = 0.0
    epochs = 20
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        
        train_loss, train_acc = train(model=model,
                                      data_loader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=DEVICE)
        
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        
        test_loss, test_acc, labels, preds = evaluate(model=model,
                                                      data_loader=test_dataloader,
                                                      loss_fn=loss_fn,
                                                      device=DEVICE)
        
        print(f'Testing Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.jit.script(model).save(MODEL_PATH)
            # torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        scheduler.step()
        print()
        
    final_acc = conf_mat(model=model,
                   data_loader=test_dataloader,
                   dataset=testset,
                   device=DEVICE,
                   model_path=MODEL_PATH,
                   confmat_path="./classification_model/model1/result/use_functions.jpg")
    
if __name__ == "__main__":
    main()
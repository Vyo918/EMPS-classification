import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import train, evaluate, conf_mat, plot_learning_curves
import os

def train_all_model(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, GRAPH_PATH, TEXT_PATH):
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
    train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True, drop_last=True)
    
    val_dir = DATA_PATH + "val"
    valset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    val_dataloader = DataLoader(valset, batch_size=32, shuffle=False)
    
    test_dir = DATA_PATH + "test"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)

    model = LeNet5(num_classes=len(testset.classes)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model.apply(init_weights)
    
    best_acc = 0
    epochs = 20
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
                os.makedirs("./classification_model/model1/model/", exist_ok=True)
                torch.jit.script(model).save(MODEL_PATH)
                flag = False
                f.write(f'Testing Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n')
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
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
    # hierarchy structure
    arr = ["big_category", "top", "bottom", "footwear"]
    acc = {"big_category": 0,
           "top": 0, 
           "bottom": 0, 
           "footwear": 0}
    for a in arr:
        print(f"Training {a} model:\n")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DATA_PATH = f"./dataset/my_dataset/model1/improved/{a}/"
        MODEL_PATH = f"./classification_model/model1/model/{a}.pt"
        CONFMAT_PATH = f"./classification_model/model1/result/confmat/{a}_train.jpg"
        GRAPH_PATH = f"./classification_model/model1/result/graph/{a}_train.jpg"
        TEXT_PATH = f"./classification_model/model1/result/text/{a}_train.txt"
        acc[a] = train_all_model(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, GRAPH_PATH, TEXT_PATH)
        print("-"*50)
        
    # without hierarchy structure
    print(f"Training model:\n")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = f"./dataset/my_dataset/model1/split/"
    MODEL_PATH = f"./classification_model/model1/model/model.pt"
    CONFMAT_PATH = f"./classification_model/model1/result/confmat/model.jpg"
    GRAPH_PATH = f"./classification_model/model1/result/graph/model.jpg"
    TEXT_PATH = f"./classification_model/model1/result/text/model.txt"
    accuracy = train_all_model(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, GRAPH_PATH, TEXT_PATH)
    print("-"*50)
    for a in arr:
        print(f"Accuracy of {a} model: {acc[a]:.2f}%")
    print(f"Accuracy of model: {accuracy:.2f}%")
if __name__ == "__main__":
    main()
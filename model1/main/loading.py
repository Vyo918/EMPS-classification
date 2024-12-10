import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            self.fc2 = nn.Linear(84, 10)
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

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dir = "./dataset/my_dataset/model1/test/"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)

    model_dir = "./classification_model/model1/best_model.pth"
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(f=model_dir))

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

    confmat = ConfusionMatrix(num_classes=len(testset.classes), task='multiclass')
    confmat_tensor = confmat(preds=y_preds, target=torch.tensor(testset.targets))

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=testset.classes,
        figsize=(10,7)
    )
    # plt.savefig("./classification_model/model1/result/best_accuracy.png")
    plt.show()
    
if __name__ == "__main__":
    main()
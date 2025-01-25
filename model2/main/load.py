import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import conf_mat, save_misclassified_images

def load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH):
    print(f"Using device: {DEVICE}")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])   
    test_dir = DATA_PATH + "test"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    final_acc, labels, preds = conf_mat(data_loader=test_dataloader,
                                        dataset=testset,
                                        device=DEVICE,
                                        model_path=MODEL_PATH,
                                        confmat_path=CONFMAT_PATH,
                                        show=True)
    
    # save_misclassified_images(data_loader=test_dataloader,
    #                           labels=labels,
    #                           preds=preds,
    #                           output_dir=MISSCLASSIFIED_PATH)
    
    return final_acc

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = f"./dataset/my_dataset/model2/split/"
    MODEL_PATH = f"./classification_model/model2/model/train(83.85).pt"
    CONFMAT_PATH = f"./classification_model/model2/result/confmat/load.jpg"
    MISSCLASSIFIED_PATH = f"./classification_model/model2/result/missclassified/load/"
    final_acc = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH)
    print(f"trained model accuracy: {final_acc:.2f}%")
        
if __name__ == "__main__":
    main()
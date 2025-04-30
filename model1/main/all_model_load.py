import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import conf_mat, save_misclassified_images

def load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold):
    print(f"Using device: {DEVICE}")

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])    
    test_dir = DATA_PATH + "test"
    testset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_dataloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # return testset.classes

    final_acc, labels, preds, below_threshold, confidences = conf_mat(data_loader=test_dataloader,
                                        dataset=testset,
                                        device=DEVICE,
                                        model_path=MODEL_PATH,
                                        confmat_path=CONFMAT_PATH,
                                        show=False,
                                        threshold=threshold)
    
    save_misclassified_images(data_loader=test_dataloader,
                              labels=labels,
                              preds=preds,
                              confidences=confidences,
                              threshold=threshold,
                            #   output_dir=MISSCLASSIFIED_PATH,
                              )
    
    return final_acc, below_threshold, len(testset)

def main():
    import os, shutil
    output_dir="./classification_model/model1/result/"
    misclassified_dir = os.path.join(output_dir, "misclassified")
    below_threshold_dir = os.path.join(output_dir, "below_threshold")
    correct_dir = os.path.join(output_dir, "correctly_classified")
    for dir in [misclassified_dir, below_threshold_dir, correct_dir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)  # Remove existing directory if exists
        os.makedirs(dir, exist_ok=True)


    arr = ["big_category", "top", "bottom", "footwear"]
    acc = {"big_category": 98.22, 
           "top": 89.22, 
           "bottom": 92.97, 
           "footwear": 92.10}
    threshold = {"big_category": 0.93, 
                "top": 0.54, 
                "bottom": 0.62, 
                "footwear": 0.57}
    for a in arr:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DATA_PATH = f"./dataset/my_dataset/model1/improved/{a}/"
        MODEL_PATH = f"./classification_model/model1/model/{a}({acc[a]}).pt"
        CONFMAT_PATH = f"./classification_model/model1/result/confmat/{a}_load.jpg"
        MISSCLASSIFIED_PATH = f"./classification_model/model1/result/missclassified/{a}_load/"
        final_acc, below_threshold, total_image = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold[a])
        print(f"{a} model accuracy: {final_acc:.2f}%")
        print(f"{below_threshold} images out of {total_image} are below threshold ({threshold[a]})")
        print()
        # print(load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH))
        
    # acc = 90.23
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DATA_PATH = f"./dataset/my_dataset/model1/split/"
    # MODEL_PATH = f"./classification_model/model1/model/model({acc}).pt"
    # CONFMAT_PATH = f"./classification_model/model1/result/confmat/model_load.jpg"
    # MISSCLASSIFIED_PATH = f"./classification_model/model1/result/missclassified/model_load/"
    # final_acc = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH)
    # print(f"model accuracy: {final_acc:.2f}%")
        
if __name__ == "__main__":
    main()
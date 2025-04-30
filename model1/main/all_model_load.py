import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import conf_mat, save_misclassified_images
import os

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

    final_acc, labels, preds, below_threshold, confidences, threshold = conf_mat(data_loader=test_dataloader,
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
    
    return final_acc, below_threshold, len(testset), threshold

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


    arr = ["big_category", "top", "bottom"]#, "footwear"]
    acc = {"big_category": 97.97, 
           "top": 88.99, 
           "bottom": 92.34}#, 
        #    "footwear": 92.10}
    threshold = {"big_category": 0.88, 
                "top": 0.51, 
                "bottom": 0.60}#, 
                # "footwear": 0.57}

    # model_path = {"big_category": f"./classification_model/model1/model/big_category(98.22, 0.9).pt",
    #                 "top": f"./classification_model/model1/model/top(, 0.53).pt", 
    #                 "bottom": f"./classification_model/model1/model/bottom(, 0.61).pt"}#, 
                    # "footwear": f"./classification_model/model1/model/footwear({acc['footwear']}).pt"}

    for a in arr:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DATA_PATH = f"./dataset/my_dataset/model1/improved_no_footwear/{a}/"
        MODEL_PATH = f"./classification_model/model1/model/{a}({acc[a]}, {threshold[a]}).pt"
        CONFMAT_PATH = f"./classification_model/model1/result/confmat/{a}_load.jpg"
        MISSCLASSIFIED_PATH = f"./classification_model/model1/result/missclassified/{a}_load/"
        final_acc, below_threshold, total_image, new_threshold = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold[a])
        print(f"{a} model accuracy: {final_acc:.5f}%")
        print(f"{below_threshold} images out of {total_image} are below threshold ({threshold[a]})")
        
        if new_threshold != threshold[a]:
            print(f"New threshold: {new_threshold:.2f}")
            threshold[a] = new_threshold
            final_acc, below_threshold, total_image, new_threshold = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold[a])
            print(f"{a} model accuracy: {final_acc:.5f}%")

            import os
            import re

            original_name = MODEL_PATH
            # Split into directory and filename
            dir_path, filename = os.path.split(original_name)

            # Apply regex to filename
            match = re.match(rf"{a}\(([^)]+)\).pt", filename)
            if match:
                values = [v.strip() for v in match.group(1).split(',')]
                values[1] = f"{new_threshold:.2f}"  # Update second value
                values.append(f"{final_acc:.2f}")  # Add new value
                
                new_inner = ", ".join(values)
                new_filename = f"{a}({new_inner}).pt"
                
                # Combine directory with new filename
                new_name = os.path.join(dir_path, new_filename)
                os.rename(original_name, new_name)  # Rename the file
                print(f"Renamed {original_name} to {new_name}")
            else:
                print("Filename pattern not matched.")


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
    # print(f"model accuracy: {final_acc:.5f}%")
        
if __name__ == "__main__":
    main()
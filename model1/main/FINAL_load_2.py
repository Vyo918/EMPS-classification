import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.functional import softmax
import os
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import hashlib
import shutil
from utils import save_misclassified_images
import os
import hashlib
from PIL import Image

def getParent(path, levels = 1):
	common = path
	for i in range(levels + 1):

		common = os.path.dirname(common)

	return os.path.relpath(path, common)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFMAT_PATH = "./classification_model/model1/result/confmat/FINAL_load.jpg"
MISSCLASSIFIED_PATH = "./classification_model/model1/result/missclassified/FINAL_load/"
DATA_PATH = "./dataset/my_dataset/model1/improved/"
DATA_NAME = {"big_category": "big_category/", 
             "top": "top/", 
             "bottom": "bottom/", 
             "footwear": "footwear/"}
for key, value in DATA_NAME.items():
    DATA_NAME[key] = os.path.join(DATA_PATH, value)
    
MODEL_PATH = "./classification_model/model1/model/"
MODEL_NAME = {"big_category": f"{MODEL_PATH}big_category.pt", 
              "top": f"{MODEL_PATH}top(90.04).pt", 
              "bottom": f"{MODEL_PATH}bottom(93.83).pt", 
              "footwear": f"{MODEL_PATH}footwear(91.07).pt"}    

def hash_tensor(tensor):
    tensor_bytes = tensor.cpu().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def get_offset(category):
    if category == "top":
        return 1
    elif category == "bottom":
        return 6
    elif category == "footwear":
        return 9
    else:
        return 0

def main():
    print(f"Using device: {DEVICE}")    
    
    test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    test_dir = {"big_category": os.path.join(DATA_NAME["big_category"], "test"), 
                "top": os.path.join(DATA_NAME["top"], "test"), 
                "bottom": os.path.join(DATA_NAME["bottom"], "test"), 
                "footwear": os.path.join(DATA_NAME["footwear"], "test")}
    
    testset = {"big_category": datasets.ImageFolder(root=test_dir["big_category"], transform=test_transform), 
               "top": datasets.ImageFolder(root=test_dir["top"], transform=test_transform), 
               "bottom": datasets.ImageFolder(root=test_dir["bottom"], transform=test_transform), 
               "footwear": datasets.ImageFolder(root=test_dir["footwear"], transform=test_transform)}
    
    test_dataloader = {"big_category": DataLoader(testset["big_category"], shuffle=False),
                       "top": DataLoader(testset["top"], shuffle=False),
                       "bottom": DataLoader(testset["bottom"], shuffle=False),
                       "footwear": DataLoader(testset["footwear"], shuffle=False)}
    
    model = {"big_category": torch.jit.load(MODEL_NAME["big_category"]), 
             "top": torch.jit.load(MODEL_NAME["top"]), 
             "bottom": torch.jit.load(MODEL_NAME["bottom"]), 
             "footwear": torch.jit.load(MODEL_NAME["footwear"])}
    
    for key, value in model.items():
        model[key] = value.to(DEVICE)
        model[key] = value.eval()
    
    LABEL = {"big_category": testset["big_category"].classes, 
             "top": testset["top"].classes, 
             "bottom": testset["bottom"].classes, 
             "footwear": testset["footwear"].classes}
    
    LABEL_DICT = {}
    for category, pathname in DATA_NAME.items():
        if category != "big_category":
            for img, lbl in test_dataloader[category]:
                img, lbl = img.to(DEVICE), lbl.to(DEVICE)
                img_hash = hash_tensor(img)
                if img_hash in LABEL_DICT:
                    print(f"Same hash detected -> {img_hash}, {LABEL_DICT[img_hash]} -> {img_hash}, {LABEL[category][lbl.item()]}")
                else:
                    LABEL_DICT[img_hash] = LABEL[category][lbl.item()]
    
    overall_correct = 0
    correct = {"big_category": 0,
               "top": 0,
               "bottom": 0,
               "footwear": 0}
    total = {"big_category": 0,
             "top": 0,
             "bottom": 0,
             "footwear": 0}
    all_labels = []
    all_preds = []
    threshold1 = 0.35
    threshold2 = 0.25
    unknown1 = []
    unknown2 = []

    with torch.no_grad():
        for images, label1 in test_dataloader["big_category"]:
            images, label1 = images.to(DEVICE), label1.to(DEVICE)
            images_hash = hash_tensor(images)

            output1 = model["big_category"](images)
            max_probs1, predicted1 = torch.max(softmax(output1, dim=1), 1)
            
            if max_probs1.item() < threshold1:
                unknown1.append((images_hash, max_probs1, pred1_category, label1_category))
            
            pred1_category = LABEL["big_category"][predicted1.item()]
            label1_category = LABEL["big_category"][label1.item()]
            
            # get the pred and label
            if pred1_category == "dress":
                pred2_category = "dress"
            else:
                output2 = model[pred1_category](images)
                max_probs2, predicted2 = torch.max(softmax(output2, dim=1), 1)
                
                if max_probs2.item() < threshold2:
                    unknown2.append((images_hash, max_probs2, pred2_category, label2_category))
            
                pred2_category = LABEL[pred1_category][predicted2.item()]
                
            if label1_category == "dress":
                label2_category = "dress"
            else:
                label2_category = LABEL_DICT.get(images_hash)
                        
            # add the pred and label to the list
            if pred1_category == "dress":
                all_preds.extend([0])
            else: 
                all_preds.extend(predicted2.cpu().numpy() + get_offset(pred1_category))
                
            if label1_category == "dress":
                all_labels.extend([0])
            else:
                all_labels.extend([(LABEL[label1_category]).index(label2_category) + get_offset(label1_category)])
                
            # count correct and total
            if pred1_category == label1_category:
                correct["big_category"] += 1
            if pred2_category == label2_category:
                overall_correct += 1
                if pred1_category != "dress":
                    correct[pred1_category] += 1
            total["big_category"] += 1
            if pred1_category != "dress":
                total[pred1_category] += 1
            
            if total["big_category"] % 100 == 0:
                print(f"Predicted: {pred1_category} -> {pred2_category}, Label: {label1_category} -> {label2_category}")
                print(f"Correct: {overall_correct}/{total['big_category']} ({100 * overall_correct / total['big_category']:.2f}%)")
            
    accuracy = {"big_category": 100 * correct["big_category"] / total["big_category"],
             "top": 100 * correct["top"] / total["top"],
             "bottom": 100 * correct["bottom"] / total["bottom"],
             "footwear": 100 * correct["footwear"] / total["footwear"]}
    overall_acc = 100 * overall_correct / total["big_category"]
            
    confmat = ConfusionMatrix(num_classes=1+len(LABEL["top"])+len(LABEL["bottom"])+len(LABEL["footwear"]), task='multiclass')
    confmat_tensor = confmat(preds=torch.tensor(all_preds), target=torch.tensor(all_labels))

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=["01Dress"]+LABEL["top"]+LABEL["bottom"]+LABEL["footwear"],
        figsize=(10,7)
    )
    
    plt.title(f"Confusion Matrix (Accuracy: {overall_acc:.2f}%)")
    plt.tight_layout()
    if CONFMAT_PATH != None:
        plt.savefig(f"{os.path.splitext(CONFMAT_PATH)[0]}({overall_acc:.2f}%){os.path.splitext(CONFMAT_PATH)[1]}")
    plt.show()
    
    
    #save missclassified images
    if os.path.exists(MISSCLASSIFIED_PATH):
        shutil.rmtree(MISSCLASSIFIED_PATH)  # Remove existing directory if exists
    os.makedirs(MISSCLASSIFIED_PATH, exist_ok=True)
    
    TRUE_LABEL = {0: "Dress",
                  1: "Shirt",
                  2: "Tshirt",
                  3: "Hoodie",
                  4: "Sweater",
                  5: "Polo Shirt",
                  6: "Pants",
                  7: "Shorts",
                  8: "Skirt",
                  9: "Flats",
                  10: "Heels",
                  11: "Shoes",
                  12: "Sneakers"}
    
    for i, (image, label, pred) in enumerate(zip(test_dataloader["big_category"], all_labels, all_preds)):
        if label != pred:
            original_image_path = testset["big_category"].samples[i][0]
            
            if os.path.exists(original_image_path):
                class_dir = os.path.join(MISSCLASSIFIED_PATH, f"True_{TRUE_LABEL[label]}_Pred_{TRUE_LABEL[pred]}")
                os.makedirs(class_dir, exist_ok=True)

                dest_path = os.path.join(class_dir, f"missclassified_{i}.png")
                shutil.copy(original_image_path, dest_path)
            else:
                print(f"can't find the path: {original_image_path}")
    
    
    for key, value in accuracy.items():
        print(f"{key}: {value:.2f}%")
    print(f"overall accuracy: {overall_acc:.2f}%")
    
    
    print(f"unknown: {len(unknown1)+len(unknown2)}")
    # print(unknown)
    
    # dataset directory where u want to find the item
    dataset = "./dataset/my_dataset/model1/improved/big_category/test"
    
    for x in [unknown1, unknown2]:
        for i in x:
            file_hash = i[0]
            probs = i[1]
            pred = i[2]
            label = i[3]
            flag = False
            for category in os.listdir(dataset):
                category_dir = os.path.join(dataset, category)
                
                for img in os.listdir(category_dir):
                    img_path = os.path.join(category_dir, img)
                    img_hash = hash_tensor(test_transform(Image.open(img_path).convert("RGB")))
                    
                    if file_hash == img_hash:
                        flag = True
                        print(f"{getParent(img_path)}: {probs}, {label} -> {pred}")
                        break
                if flag:
                    break
    
if __name__ == "__main__":
    main()
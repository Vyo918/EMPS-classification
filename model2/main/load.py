import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import conf_mat, save_misclassified_images

def load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold):
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

    acc = {
        "aug": 74.86,
        "no_boho": 72.74,
    }
    threshold = {
        "aug": 0.49,
        "no_boho": 0.46,
    }
    notes = ["aug", "no_boho"]

    for note in notes:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if note == "aug":
            DATA_PATH = f"./dataset/my_dataset/model2/split_with_augmentation/"
        else:
            DATA_PATH = f"./dataset/my_dataset/model2/split_without_boho/"
        MODEL_PATH = f"./classification_model/model2/model/train({acc[note]}, {threshold[note]}, {note}).pt"
        CONFMAT_PATH = f"./classification_model/model2/result/confmat/load.jpg"
        MISSCLASSIFIED_PATH = f"./classification_model/model2/result/missclassified/load/"
        final_acc, below_threshold, total_image, new_threshold = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold[note])
        print(f"{note} model accuracy: {final_acc:.5f}%")
        print(f"{below_threshold} images out of {total_image} are below threshold ({threshold[note]:.2f})")
        
        if new_threshold != threshold[note]:
            print(f"New threshold: {new_threshold:.2f}")
            threshold[note] = new_threshold
            final_acc, below_threshold, total_image, new_threshold = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold[note])
            print(f"{note} model accuracy: {final_acc:.5f}%")

            

            import os
            import re

            original_name = MODEL_PATH
            # Split into directory and filename
            dir_path, filename = os.path.split(original_name)

            # Apply regex to filename
            match = re.match(r"train\(([^)]+)\).pt", filename)
            if match:
                # Split the extracted values by commas and strip whitespace
                values = [v.strip() for v in match.group(1).split(',')]
                
                # Update the second value (index 1) to 0.50 and append the new value 80.02
                values[1] = f"{new_threshold:.2f}"
                values.insert(2, f"{final_acc:.2f}")
                
                # Construct the new filename
                new_inner = ", ".join(values)
                new_name = f"train({new_inner}).pt"
                
                # Rename the file
                # os.rename(filename, new_name)
                os.rename(original_name, os.path.join(dir_path, new_name))
                print(f"Renamed {filename} to {new_name}")
            else:
                print("Filename pattern not matched.")
        
        print(f"{note} model accuracy: {final_acc:.5f}%")
        print(f"{below_threshold} images out of {total_image} are below threshold ({threshold[note]:.2f})")

        print()
        # print(load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold))


if __name__ == "__main__":
    main()
    # import torch
#     from torch.nn.functional import softmax
#     import torchvision.transforms as T
#     from PIL import Image
#     import numpy as np

#     image = "backend/input/image.png"
#     image = Image.open(image).convert("RGB")
#     # image.show()
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     image = image.convert("RGB")

#     if False:
#         from torchvision.models.segmentation import deeplabv3_resnet101
#         model = deeplabv3_resnet101(pretrained=True)
#         model.eval()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = model.to(device)
#         model.eval()

#         transform = T.Compose([
#             T.Resize((520, 520)),  # Resize to match DeepLabv3+ input size
#             T.ToTensor(),          # Convert PIL Image to Tensor
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#         ])

#         original_size = image.size  # Save original size
#         input_tensor = transform(image).unsqueeze(0)
#         input_tensor = input_tensor.to(device)
        
#         with torch.no_grad():
#             output = model(input_tensor)["out"][0]
#         segmentation = output.argmax(0).byte().cpu().numpy()
#         person_mask = (segmentation == 15).astype(np.uint8) * 255

#         # Apply mask to original image with black background
#         mask_image = Image.fromarray(person_mask).resize(original_size)
#         black_bg = Image.new("RGB", original_size, (0, 0, 0))    
#         black_image = Image.composite(image, black_bg, mask_image)

#         image = black_image.convert("RGB")
#         # image.show()

#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406],
#                     [0.229, 0.224, 0.225])
#     ])   
#     LABEL = ['Bohemian', 'Casual', 'Formal', 'Semi-formal', 'Sporty', 'Streetwear']
#     MODEL_PATH = "./classification_model/model2/model/train(72.14)_aug.pt"
#     THRESHOLD = 0.41

#     model = torch.jit.load(MODEL_PATH, map_location=torch.device(device))
#     image = transform(image)
#     image = image.unsqueeze(0)
#     with torch.no_grad():
#         image = image.to(device)
#         logits = model(image)
#         prob, pred = torch.max(softmax(logits, dim=1), 1)
#         if prob.item() < THRESHOLD:
#             raise Exception("Unable to classify the outfit")
#         style = LABEL[pred.item()]

#         print(style)
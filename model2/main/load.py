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
    acc = 72.14
    threshold = 0.41
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = f"./dataset/my_dataset/model2/split_with_augmentation/"
    MODEL_PATH = f"./classification_model/model2/model/train({acc})_aug.pt"
    CONFMAT_PATH = f"./classification_model/model2/result/confmat/load.jpg"
    MISSCLASSIFIED_PATH = f"./classification_model/model2/result/missclassified/load/"
    final_acc, below_threshold, total_image = load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold)
    print(f"trained model accuracy: {final_acc:.2f}%")
    print(f"{below_threshold} images out of {total_image} are below threshold({threshold})")
    # print(load(DEVICE, DATA_PATH, MODEL_PATH, CONFMAT_PATH, MISSCLASSIFIED_PATH, threshold))


if __name__ == "__main__":
    import torch
    from torch.nn.functional import softmax
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np

    image = "backend/input/image.png"
    image = Image.open(image).convert("RGB")
    # image.show()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = image.convert("RGB")

    if False:
        from torchvision.models.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(pretrained=True)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        transform = T.Compose([
            T.Resize((520, 520)),  # Resize to match DeepLabv3+ input size
            T.ToTensor(),          # Convert PIL Image to Tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        original_size = image.size  # Save original size
        input_tensor = transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
        segmentation = output.argmax(0).byte().cpu().numpy()
        person_mask = (segmentation == 15).astype(np.uint8) * 255

        # Apply mask to original image with black background
        mask_image = Image.fromarray(person_mask).resize(original_size)
        black_bg = Image.new("RGB", original_size, (0, 0, 0))    
        black_image = Image.composite(image, black_bg, mask_image)

        image = black_image.convert("RGB")
        # image.show()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])   
    LABEL = ['Bohemian', 'Casual', 'Formal', 'Semi-formal', 'Sporty', 'Streetwear']
    MODEL_PATH = "./classification_model/model2/model/train(72.14)_aug.pt"
    THRESHOLD = 0.41

    model = torch.jit.load(MODEL_PATH, map_location=torch.device(device))
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        logits = model(image)
        prob, pred = torch.max(softmax(logits, dim=1), 1)
        if prob.item() < THRESHOLD:
            raise Exception("Unable to classify the outfit")
        style = LABEL[pred.item()]

        print(style)

    # main()
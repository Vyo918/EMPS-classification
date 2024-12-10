import os
import PIL
from PIL import Image
import cv2

# Set paths
dataset_dir = 'Classification_Model_2/dataset/FashionStyle14/dataset/'

for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    if os.path.isdir(label_dir):
        files = os.listdir(label_dir)
        num_files = len(files)
        num_trash = 0
        for file in files:
            img_path = os.path.join(label_dir, file)
            # print(file)
            try:
                img1 = Image.open(img_path)
                img2 = cv2.imread(img_path)
            except PIL.UnidentifiedImageError:
                # print(f"can't be open: {file}")
                num_trash += 1
        # print(f"{num_trash} trash out of {num_files} in {label} class")
        print(f"{num_files-num_trash} openable files out of {num_files} in {label}")
# calculate and check name of the dataset

import os

data_dir = 'C:/Users/elvio/Documents/folder_vio/Coding/College/UNDERGRADUATE-PROJECT/dataset/my_dataset/model1/'
for dataset in os.listdir(data_dir):
    dataset_dir = os.path.join(data_dir, dataset)
    num_files = 0
    print(f"{dataset} dataset: ")
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        files = os.listdir(label_dir)
        num_files += len(files)
        print(f"{len(files)} files in {label}")
    print(f"total: {num_files}")
    print()
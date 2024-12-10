import os
import shutil
import random

# Define paths
dataset_dir = 'Classification_Model_2/dataset/FashionStyle14/dataset'  # Original dataset folder containing class subfolders
output_train_dir = 'Classification_Model_2/dataset/FashionStyle14/data_80_20/train'  # Folder to store training data
output_test_dir = 'Classification_Model_2/dataset/FashionStyle14/data_80_20/test'    # Folder to store testing data
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Set the train/test split ratio
split_ratio = 0.8

# Function to split and move files
def split_data(dataset_dir, output_train_dir, output_test_dir, split_ratio):
    # Loop through each class label folder
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            # Create corresponding train/test directories for the label
            train_label_dir = os.path.join(output_train_dir, label)
            test_label_dir = os.path.join(output_test_dir, label)
            os.makedirs(train_label_dir, exist_ok=True)
            os.makedirs(test_label_dir, exist_ok=True)

            # List all files in the current label folder
            files = os.listdir(label_dir)
            files = [f for f in files if os.path.isfile(os.path.join(label_dir, f))]

            # Shuffle files randomly
            random.shuffle(files)

            # Calculate split index
            split_index = int(len(files) * split_ratio)

            # Split the files into training and testing
            train_files = files[:split_index]
            test_files = files[split_index:]

            # Move the training files
            for file in train_files:
                src_path = os.path.join(label_dir, file)
                dest_path = os.path.join(train_label_dir, file)
                shutil.copy2(src_path, dest_path)  # Use copy2 to preserve metadata (e.g., timestamps)

            # Move the testing files
            for file in test_files:
                src_path = os.path.join(label_dir, file)
                dest_path = os.path.join(test_label_dir, file)
                shutil.copy2(src_path, dest_path)

            print(f"Processed label '{label}': {len(train_files)} training files, {len(test_files)} testing files.")

# Call the function to split the data
split_data(dataset_dir, output_train_dir, output_test_dir, split_ratio)

print("Data splitting complete!")

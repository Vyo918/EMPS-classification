import os
import shutil
import random
import PIL
from PIL import Image, ImageEnhance, ImageOps

# Set paths
train_dir = 'Classification_Model_2/dataset/FashionStyle14/data_80_20/train'  # Path to your original training data
test_dir = 'Classification_Model_2/dataset/FashionStyle14/data_80_20/test'  # Path to your original testing data
output_train_dir = 'Classification_Model_2/dataset/FashionStyle14/augmented_copy_data_80_20/train'  # Path to the new output directory for augmented training data
output_test_dir = 'Classification_Model_2/dataset/FashionStyle14/augmented_copy_data_80_20/test'  # Path to the output directory for copied test data
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Number of augmented images to create for small classes
augment_count = 1000

# Function to perform data augmentation
def augment_image(img):
    # Apply a random rotation between -30 to 30 degrees
    img = img.rotate(random.uniform(-30, 30))

    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Random brightness adjustment (0.8 to 1.2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random zoom by resizing and cropping the image
    if random.random() > 0.5:
        zoom_factor = random.uniform(0.8, 1.2)
        w, h = img.size
        img = img.resize((int(w * zoom_factor), int(h * zoom_factor)), Image.Resampling.LANCZOS)
        img = img.crop((0, 0, w, h))

    # Random color adjustment (0.8 to 1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    return img

# Function to augment images for a specific class
def augment_class_images(label, augment_count, train_dir, output_train_dir):
    label_dir = os.path.join(train_dir, label)  # Folder containing images for this label
    output_label_dir = os.path.join(output_train_dir, label)  # Output folder for augmented images
    os.makedirs(output_label_dir, exist_ok=True)

    files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    existing_files_count = len(files)

    # Perform augmentation if the class has fewer than augment_count files
    for i in range(augment_count - existing_files_count):
        file = random.choice(files)
        img_path = os.path.join(label_dir, file)
        try:
            img = Image.open(img_path)
            
            # Apply augmentation
            augmented_img = augment_image(img)
            
            # Save augmented image with the correct extension
            file_name, file_ext = os.path.splitext(file)  # Split the original file name and extension
            output_path = os.path.join(output_label_dir, f'aug_{i}_{file_name}{file_ext}')  # Append the original extension
            augmented_img.save(output_path)
        except PIL.UnidentifiedImageError:
            print(f"{img_path} can't be open")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


    print(f"Augmented {label} with {augment_count - existing_files_count} new images.")

# Loop through all the labels and perform augmentation on smaller classes
for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    if os.path.isdir(label_dir):
        files = os.listdir(label_dir)
        num_files = len(files)

        # Perform augmentation for smaller classes
        if num_files < augment_count:
            augment_class_images(label, augment_count, train_dir, output_train_dir)
        else:
            # Just copy files for larger classes without augmentation
            output_label_dir = os.path.join(output_train_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)
            for file in files:
                shutil.copy2(os.path.join(label_dir, file), os.path.join(output_label_dir, file))

# Copy test data
for label in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label)
    output_label_dir = os.path.join(output_test_dir, label)
    os.makedirs(output_label_dir, exist_ok=True)
    
    for file in os.listdir(label_dir):
        shutil.copy2(os.path.join(label_dir, file), os.path.join(output_label_dir, file))

# Function to count and print the number of images for each class
def print_class_file_counts(data_dir, data_type):
    print(f"\nNumber of {data_type} images per class:")
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            num_files = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
            print(f"Class '{label}': {num_files} images")

# Print out file counts for train and test directories
print_class_file_counts(output_train_dir, "training")
print_class_file_counts(output_test_dir, "testing")

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import resize
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Example: Load the pretrained CLIPSeg model (Assuming HuggingFace model usage)
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Load and preprocess the image
image = Image.open("C:/Users/elvio/Documents/folder_vio/Coding/College/UNDERGRADUATE-PROJECT/classification_model/model2/dataset/FashionStyle14/data_csv_files/test/conservative/1efbf63023c216e9aae1a19406999383.jpg").convert("RGB")
image_inputs = processor.image_processor(image, return_tensors="pt", size={"height": 224, "width": 224})

text_prompts = ["head", "top", "bottom", "shoes"]
text_inputs = processor.tokenizer(text_prompts, return_tensors="pt", padding=True)

# Preprocess the image for the model
pixel_values = image_inputs["pixel_values"].repeat(len(text_prompts), 1, 1, 1)
inputs = {"pixel_values": pixel_values, **text_inputs}

print(inputs["pixel_values"].shape)  # Should print: [4, 3, 224, 224] (4 prompts for 1 image repeated)
print(inputs["input_ids"].shape)      # Should print: [4, seq_len]

# Generate segmentation masks
with torch.no_grad():
    outputs = model(**inputs)
    masks = outputs.logits  # Shape: [num_prompts, height, width]
    print(masks.shape)      # Should print: [4, 224, 224] (one mask per prompt)

# Postprocess the masks
# Convert logits to binary masks
binary_masks = (masks > 0).float()  # Threshold at 0 to create binary masks

num_masks = 4
# Resize masks to match feature map size (e.g., 32x32)
feature_map_size = (32, 32)
resized_masks = torch.stack([
    T.functional.resize(mask.unsqueeze(0), feature_map_size, interpolation=T.InterpolationMode.NEAREST).squeeze(0)
    for mask in binary_masks
])

# Simulate feature map from a backbone
feature_map = torch.rand(512, *feature_map_size)  # Example feature map [C, H, W]

# Apply masks to the feature map
item_features = [resized_mask * feature_map.sum(dim=0) for resized_mask in resized_masks]

# Result: `item_features` contains the isolated features for each item region.

fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))

# Plot original image
axes[0].imshow(image.transpose(1, 2, 0))  # Transpose to (H, W, C) for plotting
axes[0].set_title("Original Image")
axes[0].axis("off")

# Plot resized masks overlaid on the original image
for i, resized_mask in enumerate(resized_masks):
    axes[i + 1].imshow(image.transpose(1, 2, 0))  # Background: Original image
    axes[i + 1].imshow(resized_mask.numpy(), cmap="jet", alpha=0.5)  # Overlay: Mask
    axes[i + 1].set_title(f"Mask {i + 1} Overlay")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()
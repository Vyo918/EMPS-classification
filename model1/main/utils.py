import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def getParent(path, levels = 1):
	common = path
	for i in range(levels + 1):

		common = os.path.dirname(common)

	return os.path.relpath(path, common)

def train(model: nn.Module,
          data_loader: DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device):
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for the training dataset.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        output = model(images)
        loss = loss_fn(output, labels)
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def evaluate(model: nn.Module,
             data_loader: DataLoader,
             loss_fn: nn.Module,
             device: torch.device,
             threshold: int = 0,
             ):
    """
    Evaluate the model.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: A tuple containing the average loss, accuracy, ground truth labels, and predicted labels.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    below_threshold_count = 0
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # calculate accuracy
            prob, predicted = torch.max(softmax(outputs, dim=1), 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            all_confidences.extend(prob.cpu().numpy())  # Collect confidence scores
            valid_predictions = prob > threshold
            below_threshold_count += (~valid_predictions).sum().item()
            total += valid_predictions.sum().item()  # Count only valid predictions
            correct += (predicted[valid_predictions] == labels[valid_predictions]).sum().item()
            
            running_loss += loss.item() * images.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # PLOTTING CONFIDENCE
    import matplotlib.pyplot as plt
    import numpy as np

    plt.hist(all_confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Prediction Confidences")
    # plt.show()
    new_threshold = np.percentile(all_confidences, 5)  # Set threshold at 5th percentile
    print(f"Suggested threshold: {new_threshold:.2f}")


    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    return epoch_loss, accuracy, np.array(all_labels), np.array(all_preds), below_threshold_count, all_confidences, new_threshold


def conf_mat(data_loader: DataLoader,
             dataset: datasets.ImageFolder,
             device: torch.device,
             model_path: str,
             confmat_path: str = None,
             model: nn.Module = None,
             show=False,
             threshold: int = 0,
             ):
    """
    Generate and save the confusion matrix for the model's predictions.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        dataset (datasets.ImageFolder): Dataset containing class information.
        device (torch.device): Device to perform computations on.
        model_path (str): Path to the saved model state_dict.
        save (int): True if you want to save the confusion matrix.
        confmat_path (str): Path to save the confusion matrix plot.

    Returns:
        tuple: A tuple containing the accuracy, ground truth labels, and predicted labels.
    """
    if os.path.splitext(model_path)[1] == ".pth":
        model.load_state_dict(torch.load(model_path))
    elif os.path.splitext(model_path)[1] == ".pt":
        model = torch.jit.load(model_path)
    else:
        print(f"unknown file with file extension: {os.path.splitext(model_path)[1]}")
        
    model.to(device)
    
    _, acc, labels, preds, below_threshold, confidences, threshold= evaluate(model=model,
                                    data_loader=data_loader,
                                    loss_fn=nn.CrossEntropyLoss(),
                                    device=device,
                                    threshold=threshold,
                                    )
    
    confmat = ConfusionMatrix(num_classes=len(dataset.classes), task='multiclass')
    confmat_tensor = confmat(preds=torch.tensor(preds), target=torch.tensor(labels))

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=dataset.classes,
        figsize=(10,7)
    )
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2f}%)")
    plt.tight_layout()
    if confmat_path != None:
        plt.savefig(f"{os.path.splitext(confmat_path)[0]}({acc:.2f}%){os.path.splitext(confmat_path)[1]}")
    if show:
        plt.show()

    return acc, labels, preds, below_threshold, confidences, threshold

def save_misclassified_images(data_loader: DataLoader,
                              labels,
                              preds,
                              confidences,
                              threshold=0,
                              output_dir="./classification_model/model1/result/"):
    """
    Save all misclassified images to the specified output directory.

    Args:
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        dataset (Dataset): Dataset to retrieve image paths or tensors.
        dataset_path (str): Path to the test dataset.
        labels (list): Ground truth labels.
        preds (list): Model predictions.
        output_dir (str): Directory to save misclassified images.

    Returns:
        None
    """
    misclassified_dir = os.path.join(output_dir, "misclassified")
    below_threshold_dir = os.path.join(output_dir, "below_threshold")
    correct_dir = os.path.join(output_dir, "correctly_classified")
    for dir in [misclassified_dir, below_threshold_dir, correct_dir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)  # Remove existing directory if exists
        os.makedirs(dir, exist_ok=True)
    
    for i, ((image, _), label, pred, conf) in tqdm(enumerate(zip(data_loader.dataset, labels, preds, confidences)), desc="Saving Images"):
        original_image_path = data_loader.dataset.samples[i][0]

        # Skip if file doesn't exist
        if not os.path.exists(original_image_path):
            print(f"Can't find the path: {original_image_path}")
            continue
        
        # Define category-based directories
        true_label = data_loader.dataset.classes[label]
        pred_label = data_loader.dataset.classes[pred]

        if label != pred:
            class_dir = os.path.join(misclassified_dir, f"True_{true_label}_Pred_{pred_label}")
        elif conf < threshold:
            class_dir = os.path.join(below_threshold_dir, true_label)
        else:
            class_dir = os.path.join(correct_dir, pred_label)
            # continue  # Correctly classified & above threshold, no need to save

        os.makedirs(class_dir, exist_ok=True)

        dest_path = os.path.join(class_dir, f"image_{i}.png")
        shutil.copy(original_image_path, dest_path)

def plot_learning_curves(history,
                         acc,
                         graph_path=None,
                         show=False):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    if graph_path != None:
        plt.savefig(f"{os.path.splitext(graph_path)[0]}({acc:.2f}%){os.path.splitext(graph_path)[1]}")
    if show:
        plt.show()
# evaluate the test data
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import torch.nn as nn
from glob import glob
from PIL import Image
from torchvision import transforms
from model import ModelFactory


class Config:
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Update this path if different
    model_dir = "/home/sonujha/rnd/LeafDiseaseClassification/models/"
    submission_file = 'submission.csv'


class LeafDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.images = sorted(glob(os.path.join(data_dir, "*")))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        # Return image path for mapping predictions later
        return img, os.path.basename(img_path)


def load_model(model_class, num_classes, model_path, device):
    """
    Load a model with the given architecture and weights.

    Args:
        model_class: The class of the model (e.g., resnet18).
        num_classes: Number of output classes.
        model_path: Path to the model weights.
        device: Device to load the model on.

    Returns:
        The loaded model in evaluation mode.
    """
    model = model_class(pretrained=False)
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    # Adjust the state_dict if it's wrapped in 'state_dict' key
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_ensemble_predictions(models, dataloader, device):
    """
    Get ensemble predictions by averaging logits from multiple models.

    Args:
        models: List of loaded models.
        dataloader: DataLoader for the test dataset.
        device: Device to perform computations on.

    Returns:
        List of predicted class indices.
    """
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            imgs, _ = batch
            imgs = imgs.to(device)
            batch_logits = []
            for model in models:
                outputs = model(imgs)  # Shape: (batch_size, num_classes)
                batch_logits.append(outputs)
            # Stack logits from all models: Shape (num_models, batch_size, num_classes)
            stacked_logits = torch.stack(batch_logits)
            # Average logits: Shape (batch_size, num_classes)
            avg_logits = torch.mean(stacked_logits, dim=0)
            # Get predicted class indices
            _, preds = torch.max(avg_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds


def main():
    # Initialize configuration
    config = Config()

    # Define paths
    data_dir = "data/"
    test_image_path = os.path.join(data_dir, 'test_images')
    sample_submission_path = os.path.join(data_dir, 'sample_submission.csv')

    # Prepare test dataset and dataloader
    test_dataset = LeafDataset(test_image_path, transforms=config.valid_tfms)
    test_dl = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model paths
    model_paths = glob(config.model_dir+"*.pth")
    print(f"models are {model_paths}")

    # Verify that both model files exist
    for path in model_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    # Load both models
    models = []
    for path in model_paths:
        print(f"Loading model from {path}...")
        model = ModelFactory.get_model(config)
        models.append(model)
    print("All models loaded successfully.")

    # Get ensemble predictions
    print("Starting ensemble predictions...")
    preds = get_ensemble_predictions(models, test_dl, config.device)
    print("Ensemble predictions completed.")

    # Prepare submission
    print("Preparing submission file...")
    sample_df = pd.read_csv(sample_submission_path)
    # Ensure that the number of predictions matches the number of test images
    test_images = sorted(os.listdir(test_image_path))
    if len(preds) != len(test_images):
        raise ValueError(
            "Number of predictions does not match number of test images.")

    # Assign predictions to the sample submission dataframe
    sample_df['label'] = preds
    # Optionally, ensure 'image_id' matches the test images
    # If 'sample_submission.csv' already has 'image_id', ensure correct order
    # Otherwise, you might need to set 'image_id' explicitly
    # Here, we assume the order matches
    # If not, uncomment the following line:
    # sample_df['image_id'] = test_images

    # Save submission
    sample_df.to_csv(config.submission_file, index=False)
    print(f"Submission file saved to {config.submission_file}")
    print(sample_df.head())


if __name__ == "__main__":
    main()


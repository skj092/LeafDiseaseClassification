# main.py

import torch
from dataset import LeafDataset
from torch.utils.data import DataLoader, Subset
from model import ModelFactory
from config import Config
import torch.nn as nn
import numpy as np
from engine import train_one_epoch
import argparse
from create_fold import get_fold
from pathlib import Path
import wandb
from dotenv import load_dotenv
import json
import os
import sys
import logging
from datetime import datetime



# Initialize environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log")
    ]
)
logger = logging.getLogger(__name__)


def run_experiment(config: Config):
    """
    Runs a single experiment based on the provided configuration.
    """
    try:
        # Override configuration based on environment
        config.override_with_env(config.env)
        # Initialize W&B
        wandb.login()
        # Extract name from path
        experiment_name = f"{Path(config.MODEL_PATH).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="cassava-experiment",
                   config=vars(config), name=experiment_name)

        # Prepare dataset and dataloader
        df = get_fold(config.data_path)

        for fold in range(config.n_fold):
            train_df = df[df.kfold != 0].reset_index(drop=True)
            valid_df = df[df.kfold == 1].reset_index(drop=True)

            train_ds = LeafDataset(train_df, config.data_path,
                                   transforms=config.train_tfms)
            valid_ds = LeafDataset(valid_df, config.data_path,
                                   transforms=config.valid_tfms)

            if config.env == 'local':
                train_ds = Subset(train_ds, np.arange(10))
                valid_ds = Subset(valid_ds, np.arange(2))

            train_dl = DataLoader(
                train_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
            valid_dl = DataLoader(
                valid_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )

            # Model setup
            model = ModelFactory.get_model(config)
            model = model.to(config.device)

            # Loss and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            # Training loop
            for epoch in range(config.num_epochs):
                train_loss, valid_loss, train_acc, valid_acc = train_one_epoch(
                    model, train_dl, valid_dl, loss_fn, optimizer)
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_accuracy": train_acc,
                    "valid_accuracy": valid_acc
                })
                logger.info(f"Experiment: {experiment_name} | Epoch [{epoch+1}/{config.num_epochs}] - "
                            f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                            f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")

            # Save the model
            model_dir = Path(config.MODEL_PATH)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = model_dir / f'model_fold_{fold}.pth'
            torch.save(model.state_dict(), model_save_path)

        wandb.finish()
        logger.info(f"Experiment {experiment_name} completed successfully.")

    except Exception as e:
        logger.error(e)
        logger.error(f"Experiment {config.MODEL_PATH} failed with error: {e}")
        wandb.finish()
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Experiment Runner")
    parser.add_argument('--config_dir', type=str, required=True,
                        help='Path to the directory containing config files')
    parser.add_argument('--env', type=str,
                        choices=['local', 'kaggle'], required=True,
                        help='Environment to run the experiments in')
    args = parser.parse_args()

    # Validate config directory
    if not os.path.isdir(args.config_dir):
        logger.error(f"Config directory {args.config_dir} does not exist.")
        sys.exit(1)

    # List all config files in the config directory
    config_files = [f for f in os.listdir(
        args.config_dir) if f.endswith('.json')]

    if not config_files:
        logger.error(f"No configuration files found in {args.config_dir}")
        sys.exit(1)

    # Iterate over each config file and run the experiment
    for config_file in config_files:
        config_path = os.path.join(args.config_dir, config_file)
        logger.info(f"Starting experiment with config: {config_path}")

        # Load configuration
        config = Config.from_json(config_path)
        config.env = args.env  # Set the environment

        # Run the experiment
        try:
            run_experiment(config)
        except Exception as e:
            logger.error(
                f"Experiment with config {config_file} failed. Skipping to next. {e}")
            continue  # Proceed to the next experiment


if __name__ == "__main__":
    main()


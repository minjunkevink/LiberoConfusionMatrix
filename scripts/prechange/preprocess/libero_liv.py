import hydra
from omegaconf import DictConfig
from pathlib import Path
import wandb
import matplotlib.pyplot as plt


import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../config/data", config_name="libero", version_base="1.2")
def main(cfg: DictConfig):
    print("Loaded Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Example: Access dataset paths
    print(f"Processing dataset: {cfg.data.name}")
    for dataset_name, dataset_path in cfg.data.datasets.items():
        print(f" - {dataset_name}: {dataset_path}")

    # Your custom logic here
    print("Running Libero LIV pipeline...")
    # Add any specific processing code for Libero LIV here.

if __name__ == "__main__":
    main()
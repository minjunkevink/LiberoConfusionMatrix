import os
import glob
import json
import h5py
# import cv2
import numpy as np
import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../../config/data", config_name="libero")
def main(cfg: DictConfig):
    # Extract the libero_spatial dataset path from the config
    libero_spatial_dir = cfg.data.datasets.libero_spatial

    # Output directory for processed data
    output_dir = "./data/libero_spatial_processed"
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.json")
    samples = []

    # Iterate over each HDF5 file
    hdf5_files = glob.glob(os.path.join(libero_spatial_dir, "*.hdf5"))
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, "r") as f:
            # Extract language instruction
            if "language" in f:
                instruction = f["language"][()]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode('utf-8')
            else:
                print(f"Warning: No language found in {hdf5_file}, skipping.")
                continue

            # Extract final frame
            if "observations" in f and "rgb" in f["observations"]:
                rgb = f["observations"]["rgb"][:]  # load all frames
                final_frame = rgb[-1]  # last frame
            else:
                print(f"Warning: No rgb data found in {hdf5_file}, skipping.")
                continue

            # Save the final frame as an image
            base_name = os.path.splitext(os.path.basename(hdf5_file))[0]
            img_name = f"{base_name}_final.jpg"
            img_path = os.path.join(output_dir, img_name)

            # Assuming final_frame is BGR or RGB; adjust if needed
            cv2.imwrite(img_path, final_frame)

            samples.append({
                "image_path": img_path,
                "instruction": instruction
            })

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)

    print("Preprocessing complete. Manifest and images saved in:", output_dir)

    # (Optional) Run LIV on the processed dataset
    # You may need to adjust the LIV command and configs.
    # For example:
    # subprocess.run([
    #     "python", "LIV/liv/train_liv.py",
    #     "data=libero_spatial_processed",
    #     "training=finetune"
    # ], check=True)

    # After embedding extraction, compute confusion matrix (pseudo-code):
    # embeddings = load_liv_embeddings(output_dir)
    # instructions_vec = embeddings['text']
    # images_vec = embeddings['image']
    # # Compute similarity matrix (e.g., cosine similarity)
    # confusion_matrix = compute_confusion_matrix(instructions_vec, images_vec)
    # np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), confusion_matrix)

if __name__ == "__main__":
    main()
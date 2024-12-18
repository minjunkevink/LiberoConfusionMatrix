import os
import glob
import json
import h5py
import cv2
import numpy as np

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # Extract the libero_spatial dataset path from the config
    libero_spatial_dir = cfg.data.datasets.libero_spatial
    
    import ipdb; ipdb.set_trace()
    # TODO: model-specific preprocessing pipeline
    output_dir = cfg.models.liv.manifest_output_dir
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.json")
    samples = []

    # Iterate over each HDF5 file in libero_spatial
    hdf5_files = glob.glob(os.path.join(libero_spatial_dir, "*.hdf5"))
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, "r") as f:
            # Extract language instruction
            # Adjust keys if needed based on actual dataset structure
            if "language" in f:
                instruction = f["language"][()]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode('utf-8')
            else:
                print(f"Warning: No language found in {hdf5_file}, skipping.")
                continue

            # Extract final frame
            # Assuming the structure: f["observations"]["rgb"] = (T, H, W, C)
            if "observations" in f and "rgb" in f["observations"]:
                rgb = f["observations"]["rgb"][:]
                final_frame = rgb[-1]  # last frame
            else:
                print(f"Warning: No rgb data found in {hdf5_file}, skipping.")
                continue

            # Save the final frame as an image
            base_name = os.path.splitext(os.path.basename(hdf5_file))[0]
            img_name = f"{base_name}_final.jpg"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, final_frame)

            # Add entry to samples
            samples.append({
                "image_path": img_path,
                "instruction": instruction
            })

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"Preprocessing complete. Manifest and images saved in: {output_dir}")
    print("Next, you can run LIV or other models using the generated manifest.")

if __name__ == "__main__":
    main()
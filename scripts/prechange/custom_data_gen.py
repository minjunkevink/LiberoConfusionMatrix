import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import h5py
import json
from sentence_transformers import SentenceTransformer

imwidth = 84
imheight = 84

base_path = "/scr/kimkj/LIBERO/libero/datasets/"
model = SentenceTransformer("all-MiniLM-L6-v2")

splits = ["libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"]

# Ensure the output directory exists
output_dir = os.path.join(base_path, "processed_libero_dataset_lowres")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for type_split in splits:
    file_paths = os.path.join(base_path, type_split)
    output_split_dir = os.path.join(output_dir, type_split)
    if not os.path.exists(output_split_dir):
        os.mkdir(output_split_dir)

    files = os.listdir(file_paths)
    files = [file for file in files if "hdf5" in file]

    for file in tqdm(files, desc=f"Processing {type_split}"):
        curr_path = os.path.join(file_paths, file)
        output_file = os.path.join(output_split_dir, f"processed_{file}")

        # Open the source HDF5 file
        with h5py.File(curr_path, "r") as scene_demos:
            lang_instr = json.loads(scene_demos["data"].attrs["problem_info"])[
                "language_instruction"
            ]
            embedded_lang = model.encode([lang_instr])

            # Open the target HDF5 file for writing
            with h5py.File(output_file, "w") as output_h5:
                for demo in scene_demos["data"].keys():
                    demo_group = output_h5.create_group(demo)

                    # Load actions and terminals
                    actions = scene_demos["data"][demo]["actions"][()]
                    terminals = scene_demos["data"][demo]["dones"][()][:, None]

                    # Resize rendered frames if necessary
                    rendered_frames = scene_demos["data"][demo]["obs"]["agentview_rgb"][
                        ()
                    ]
                    if (
                        rendered_frames.shape[1] != imwidth
                        or rendered_frames.shape[2] != imheight
                    ):
                        rendered_frames = np.stack(
                            [
                                np.array(
                                    Image.fromarray(frame).resize((imwidth, imheight))
                                )
                                for frame in rendered_frames
                            ]
                        ).astype(np.uint8)

                    # Flip the frames along the X-axis
                    rendered_frames = rendered_frames[:, ::-1, :, :]

                    # Save the processed data to the new HDF5 file
                    demo_group.create_dataset("actions", data=actions.astype(np.float32))
                    demo_group.create_dataset("terminals", data=terminals.astype(np.uint8))
                    demo_group.create_dataset(
                        "rendered_frames", data=rendered_frames.astype(np.uint8)
                    )
                    demo_group.create_dataset(
                        "lang_embeds",
                        data=np.tile(
                            embedded_lang.astype(np.float16), (actions.shape[0], 1)
                        ),
                    )
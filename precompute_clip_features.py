#Compute CLIP vision features for the dataset and save to disk for faster loading during training. This is especially important if the dataset is large and computing CLIP features on the fly would be too slow. The saved features can be loaded during training to speed up the data pipeline.

import os
import json
import numpy as np
import jax.numpy as jnp
from PIL import Image


def precompute_clip_features(
    clip_bundle,
    tokenized_json,
    image_root,
    output_dir,
):
    with open(tokenized_json, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(data):
        image_path = os.path.join(image_root, sample["image"])
        img = Image.open(image_path).convert("RGB")

        clip_inputs = clip_bundle.processor(images=img, return_tensors="np")
        pixel_values = jnp.array(clip_inputs["pixel_values"])

        vision_outputs = clip_bundle.model.vision_model(pixel_values=pixel_values)
        vision_feats = np.array(vision_outputs.last_hidden_state[0])  # [N_vis, D_clip]

        save_path = os.path.join(output_dir, sample["image"].replace(".jpg", ".npy"))
        np.save(save_path, vision_feats)
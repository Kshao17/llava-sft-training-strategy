def build_stage1_manifest(tokenized_json, feature_dir, output_json):
    '''Build a manifest JSON file for stage 1 training. The manifest will contain the paths to the precomputed CLIP vision features as well as the tokenized input ids and labels for each sample. This manifest can then be used by the data loader during training to efficiently load the necessary data for each sample.'''
    import json
    import os

    with open(tokenized_json, "r") as f:
        data = json.load(f)

    manifest = []
    for sample in data:
        manifest.append({
            "vision_path": os.path.join(
                feature_dir,
                sample["image"].replace(".jpg", ".npy")
            ),
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
        })

    with open(output_json, "w") as f:
        json.dump(manifest, f)
import json
import os

COCO_JSON = "../data/raw/annotations/captions_train2017.json"

SAVE_PATH = "../data/processed/stage1_alignment/alignment.json"

with open(COCO_JSON) as f:
    coco = json.load(f)

id_to_filename = {
    img["id"]: img["file_name"]
    for img in coco["images"]
}

dataset = []

for ann in coco["annotations"]:
    dataset.append({
        "image": id_to_filename[ann["image_id"]],
        "caption": ann["caption"]
    })

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

with open(SAVE_PATH, "w") as f:
    json.dump(dataset, f, indent=2)
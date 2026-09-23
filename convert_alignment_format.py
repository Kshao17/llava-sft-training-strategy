import json
import random

PROMPTS = [
    "Describe this image briefly.",
    "What is in this image?",
    "Provide a short description of the image.",
    "Summarize this picture."
]

with open("../data/processed/stage1_alignment/alignment.json") as f:
    data = json.load(f)

formatted = []

for sample in data:
    formatted.append({
        "image": sample["image"],
        "instruction": random.choice(PROMPTS),
        "response": sample["caption"]
    })

with open("../data/processed/stage1_alignment/alignment_chat.json", "w") as f:
    json.dump(formatted, f, indent=2)
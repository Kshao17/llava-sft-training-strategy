import os
import subprocess
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from transformers import AutoProcessor, FlaxCLIPVisionModel


@dataclass
class ClipBundle:
    model: FlaxCLIPVisionModel
    processor: AutoProcessor
    hidden_size: int
    image_size: int
    patch_size: int
    model_dir: str


def download_clip_flax(
    repo_id: str = "openai/clip-vit-base-patch32",
    local_dir: str = "/home/hf_models/clip-vit-base-patch32",
) -> str:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.makedirs(local_dir, exist_ok=True)

    cmd = [
        "hf", "download", repo_id,
        "--include",
        "flax_model.msgpack",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "--local-dir", local_dir,
    ]
    subprocess.run(cmd, check=True)
    return local_dir


def load_clip_flax_local(
    local_dir: str = "/home/hf_models/clip-vit-base-patch32",
    dtype=jnp.bfloat16,
) -> ClipBundle:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    processor = AutoProcessor.from_pretrained(
        local_dir,
        local_files_only=True,
    )

    model = FlaxCLIPVisionModel.from_pretrained(
        local_dir,
        local_files_only=True,
        dtype=dtype,
    )

    vision_cfg = model.config

    return ClipBundle(
        model=model,
        processor=processor,
        hidden_size=vision_cfg.hidden_size,
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        model_dir=local_dir,
    )


def build_clip_vision_tower(
    local_dir: str = "/home/hf_models/clip-vit-base-patch32",
    dtype=jnp.bfloat16,
) -> ClipBundle:
    return load_clip_flax_local(local_dir=local_dir, dtype=dtype)


def create_clip_from_flax_checkpoint(
    local_dir: str = "/home/hf_models/clip-vit-base-patch32",
    download_if_missing: bool = True,
    dtype=jnp.bfloat16,
):
    if download_if_missing and not os.path.exists(os.path.join(local_dir, "flax_model.msgpack")):
        download_clip_flax(local_dir=local_dir)
    return load_clip_flax_local(local_dir=local_dir, dtype=dtype)
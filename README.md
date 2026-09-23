# LLaVA SFT Training Strategy Optimization

This project explores training strategies for multimodal supervised fine-tuning (SFT), inspired by the LLaVA visual instruction tuning framework.

The project focuses on improving training behavior under fixed data and compute constraints by comparing multiple optimization strategies, including curriculum learning, dynamic loss re-weighting, and influence-guided gradient reweighting.

## Project Overview

This was a team-based course project focused on reproducing and extending ideas from *Visual Instruction Tuning (LLaVA)*.

Our group was responsible for the **SFT training strategy** component. We investigated how different training strategies affect convergence, optimization stability, and validation loss in multimodal instruction tuning.

## My Contribution

My primary responsibility was exploring and evaluating SFT training strategies, including:

- Curriculum Learning
- Dynamic Loss Re-weighting
- Influence-guided / Gradient-based Reweighting
- Multi-stage Supervised Fine-Tuning

## Methods

### 1. Curriculum Learning

Training samples were organized by difficulty so that the model could progress from simpler tasks to more complex reasoning tasks.

The curriculum included stages such as:

- Perception
- Description
- Reasoning

This approach was designed to provide more structured training signals during the early stages of optimization.

### 2. Dynamic Loss Re-weighting

Instead of treating all training signals equally, higher importance was assigned to reasoning-related information.

Reasoning-heavy tokens and samples received larger weights so that the model could focus more on meaningful reasoning patterns during training.

### 3. Influence-guided Reweighting

Gradient agreement was used as an optimization signal.

Samples whose gradients aligned with useful validation directions were assigned greater importance, while conflicting gradients were down-weighted.

This strategy was designed to reduce noisy or contradictory training signals.

## Experimental Results

We compared four training configurations:

| Method | Train Loss | Validation Loss |
| --- | ---: | ---: |
| Baseline | 0.0274 | 0.00080 |
| Curriculum Learning | 0.0325 | 0.00080 |
| Curriculum + Dynamic Loss Re-weighting | 0.0242 | 0.00071 |
| Influence-guided Reweighting | 0.0269 | 0.00065 |

The experiments showed that:

- Curriculum learning improved early-stage convergence.
- Curriculum learning combined with dynamic loss re-weighting achieved better final optimization than curriculum learning alone.
- Influence-guided reweighting achieved a lower validation loss in the reported experiments while using reduced training computation.
- Training strategy improved optimization behavior, although data quality and composition remained important factors.

## Model Setup

The experimental setup used:

- Llama 3.2 1B Instruct
- LoRA fine-tuning
- Rank: 8
- Alpha: 32
- Multimodal visual-language training
- CLIP visual features

## Technologies

- Python
- JAX
- Flax
- CLIP
- Llama 3.2
- LoRA
- Supervised Fine-Tuning
- Multimodal Learning

## Repository Contents

- `LLaVA_Public.ipynb`  
  Main notebook containing the multimodal training and experimental workflow.

- `build_stage1_manifest.py`  
  Builds training manifests containing precomputed vision features, tokenized inputs, and labels.

- `clip_helpers.py`  
  Utilities for loading the CLIP vision encoder.

- `convert_alignment_format.py`  
  Converts image-caption alignment data into instruction-response format.

- `precompute_clip_features.py`  
  Precomputes CLIP vision features to improve training data-loading efficiency.

- `prepare_stage1_dataset.py`  
  Prepares image-caption alignment data for Stage 1 training.

- `model.py` and `params.py`  
  Llama model and parameter-loading utilities used by the training pipeline.

## Notes

This repository is based on a team course project. My primary contribution was the **SFT training strategy and experimental comparison** component.

Some model implementation files are adapted from the Google/Tunix codebase and retain their original Apache 2.0 copyright and license notices.

Large datasets, model checkpoints, credentials, and private training artifacts are not included in this repository.

# LLaVA SFT Training Strategy Optimization

Course team project at Brandeis University that reproduces a small-scale version of
[Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) in JAX/Flax and compares
training strategies for the supervised fine-tuning (SFT) stage.

The project was split across six groups. Our group (Group 3) studied **how the order and weighting
of training examples affect multimodal SFT**.

## My Contribution

- Implemented the **task-weighted loss** used in the re-weighting experiments: per-example weights
  by task type, combined with target-token masking so that only answer tokens contribute to the loss
  (`train_step_stage2` in `LLaVA_Public.ipynb`).
- Ran and analyzed the comparison of training strategies against the shared baseline.

## Pipeline

Training follows LLaVA's two stages, at a smaller scale:

| Stage | What is trained | Data | Purpose |
| --- | --- | --- | --- |
| 1. Feature alignment | Vision projector only (CLIP and Llama frozen) | COCO 2017 captions | Map CLIP image features into the language model's embedding space |
| 2. Instruction tuning (SFT) | Vision projector and language model | LLaVA-style conversation, detailed-description and complex-reasoning examples on COCO images | Teach the model to follow instructions about images |

- **Language model:** Llama 3.2 1B Instruct (JAX implementation adapted from Google's [Tunix](https://github.com/google/tunix))
- **Vision encoder:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), frozen; features are precomputed once
- **Projector:** single linear layer from CLIP features to the Llama embedding dimension
- **Train / validation split:** by image, 1,000 validation images, split seed 42

## Training Strategies Compared (Stage 2)

| Strategy | Description |
| --- | --- |
| Baseline | Cross-entropy averaged over answer tokens, examples in shuffled order |
| Curriculum learning | Examples ordered by task difficulty: conversation → detailed description → complex reasoning |
| Curriculum + loss re-weighting | Curriculum order plus a weighted loss that gives complex-reasoning examples a higher weight |
| Influence-guided reweighting | Examples up-weighted when their gradient aligns with a validation gradient direction, down-weighted when it conflicts; run under a reduced compute budget |

The weighted loss is a weighted average over masked answer tokens, so re-weighting changes the
relative contribution of each task without changing the overall scale of the loss:

```
L = sum_i w_t(i) * sum_j m_ij * l_ij  /  sum_i w_t(i) * sum_j m_ij
```

where `l_ij` is the token cross-entropy, `m_ij` the answer-token mask and `w_t(i)` the weight for
the task type of example `i`.

## Results

| Method | Train loss | Validation loss |
| --- | ---: | ---: |
| Baseline | 0.0274 | 0.00080 |
| Curriculum learning | 0.0325 | 0.00080 |
| Curriculum + loss re-weighting | 0.0242 | 0.00071 |
| Influence-guided reweighting | 0.0269 | 0.00065 |

Values are those reported in the team's final report. The JSON files in `results/` are the raw
training logs, with per-step training loss for each run.

**Observations**

- Curriculum learning reduced training loss faster early on but ended level with the baseline.
- Adding loss re-weighting to the curriculum gave a lower validation loss than the baseline.
- Influence-guided reweighting reached the lowest validation loss despite its reduced compute budget.

**Limitations**

- Each configuration was run once, so differences have no estimate of run-to-run variation.
- Only loss was measured; no benchmark of answer quality or reasoning accuracy was run.
- Training losses are not directly comparable across methods, because weighted and reordered
  losses measure different quantities from the baseline's.

## Repository Structure

```text
llava-sft-training-strategy/
├── README.md
├── requirements.txt
├── LLaVA_Public.ipynb                     # full pipeline: data prep, Stage 1, Stage 2, evaluation
├── results/
│   ├── group3_baseline_results.json
│   ├── group3_curriculum_results.json
│   ├── group3_curriculum_weight_results.json
│   ├── group3_dynamic_loss_reweighting_results.json
│   └── group3_influence_results.json
└── src/
    ├── data/
    │   ├── build_stage1_manifest.py
    │   ├── convert_alignment_format.py
    │   └── prepare_stage1_dataset.py
    ├── model/
    │   ├── model.py                       # Llama model, adapted from Tunix (Apache 2.0)
    │   └── params.py
    └── vision/
        ├── clip_helpers.py
        └── precompute_clip_features.py
```

## Running the Project

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download COCO 2017 train images and caption annotations, and set up access to Llama 3.2 1B
   Instruct (Kaggle or Hugging Face credentials).
3. Run `LLaVA_Public.ipynb` from top to bottom. The notebook prepares the data, precomputes CLIP
   features, trains Stage 1, then runs the Stage 2 experiments. A GPU or TPU is required.

Datasets, model checkpoints and credentials are not included in this repository.

## References

- Liu, H., Li, C., Wu, Q., and Lee, Y. J. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). NeurIPS 2023.
- Bengio, Y., Louradour, J., Collobert, R., and Weston, J. (2009). Curriculum Learning. ICML 2009.
- Google [Tunix](https://github.com/google/tunix): JAX library for LLM post-training (model code adapted under Apache 2.0).

## License Notes

Files in `src/model/` are adapted from Tunix and keep their original Apache 2.0 notices.

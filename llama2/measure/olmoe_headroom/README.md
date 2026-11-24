# OLMoE headroom check

Measure aligned OLMoE, full PureBad-100 fine-tuning, and official Safe Delta on
the same HEx-PHI-330 and MMLU-200 panels as the Llama-2 experiment. The model
is `allenai/OLMoE-1B-7B-0125-Instruct`. Preflight pins its Hub revision. The
SafeMoE SQL result of 33 used q/v LoRA, a different training mixture, JailbreakBench
and Llama Guard 4. This experiment uses the PureBad protocol and Guard 3.

## Machines and storage

Use a single 80 GB GPU for full BF16 AdamW training. A microbatch of one
preserves the effective batch by accumulating the token-weighted gradients of
each reference batch. Single-rank training uses native gradient checkpointing
and the reference epoch-wise StepLR. A single GPU needs no distributed sharding.
The aligned model and training compute use BF16. We export the fine-tuned
and repaired models to FP16. Calibration and repair target 24 GB VRAM and
22 GB host RAM. These hardware estimates require validation on a real run.
Allow at least 100 GiB of persistent free disk plus space for Guard and the Hub
cache. Use a shared persistent run directory or copy it between machines after
the producing command exits. Repeat shell setup in each new terminal.

## 1. Get the branch and activate the environment

Use your existing CUDA-enabled `safedelta-cu128` environment with Transformers
4.46.3. On a fresh machine, restore the training environment and confirm Torch
supports the GPU. Set HF authentication for the same gated Guard model as before.
For a vLLM reference protocol, use the original working vLLM environment/version.

```bash
cd /workspace
git clone --branch safe-delta-moe --single-branch https://github.com/kiarashsedghigh/SafeDelta.git SafeDelta-moe
cd /workspace/SafeDelta-moe/llama2
source /root/miniforge3/etc/profile.d/conda.sh
conda activate safedelta-cu128
export CUDA_VISIBLE_DEVICES=0 HF_HUB_ENABLE_HF_TRANSFER=0
RUN=runs/olmoe-0125/purebad100/42
MOE=(python -m measure.olmoe_headroom --run "$RUN")
python -m unittest -v measure.olmoe_headroom.test_headroom
```

## 2. Export the original protocol once

Only the previous evaluation result, frozen MMLU item file and registration
manifest are read. The manifest normally embeds the training YAML. If it
does not, append `--train-params /path/to/train_params.yaml`. The command
checks the PureBad data hash and the item hash recorded by the prior scorer.
When several evaluations exist, append `--results /path/to/results.json`
for the completed reference comparison. The resulting JSON is portable.

```bash
OLD=/workspace/SafeDelta/llama2/runs/llama2-7b-chat/purebad100/42
python -m measure.olmoe_headroom export-reference \
  --reference-run "$OLD" --output /workspace/olmoe-reference.json
"${MOE[@]}" preflight --reference /workspace/olmoe-reference.json --scale 0.11
```

Preflight also stores 128 calibration row IDs, actual source hashes, the Guard
revision, decoding settings and the list of all target weight names. Torch,
Transformers, Accelerate, NumPy and safetensors versions are recorded. Use the
same software versions on both machines. Exporting the reference again is
unnecessary once you have the JSON.

## 3. Aligned, fine-tuned and repaired checkpoints

```bash
"${MOE[@]}" evaluate --arm aligned
# Use the larger GPU for the next command.
"${MOE[@]}" train --microbatch 1
"${MOE[@]}" evaluate --arm no-defense
"${MOE[@]}" calibrate
"${MOE[@]}" repair
"${MOE[@]}" audit
"${MOE[@]}" evaluate --arm safedelta
"${MOE[@]}" table
```

Each command must finish successfully before its dependent stage. On another
machine, the shell variables must point to the copied run. The Hub snapshot
is fetched by its pinned revision if absent from that machine. If supplied a
completed stage, the launcher uses its cache. Calibration and repair resume by
block; interrupted training restarts from the base checkpoint. Generation
publishes per completed arm, Guard and utility predictions per item.

## What is measured

All 16 blocks contribute four attention projections, the router, and three
projections for each of 64 experts. Calibration uses native, unpadded OLMoE
conversations and aligned block outputs before repair. We record the Gram of
the input each linear layer actually received. Gate and up share a Gram only
after checking that their inputs alias. Down uses the intermediate expert
activation. Router coverage must equal the number of real tokens in the block
and total expert assignments must equal eight times that count. The pinned
implementation calls even empty experts. We keep Safe Delta's call normalization
and report nonempty calls and actual routed tokens separately.

The existing SafeDeltaRunner performs dead-diagonal handling, 1% damping,
selection and four-column compensation. Experts with positive counts below the
input width are flagged. A zero-token expert with a nonzero tuned delta raises
an error. Other weights come from the aligned model. The budget audit reloads
the saved model and computes the damped quadratic per module using half the sum
of recorded tile cutoffs as the allowance. It does not change the checkpoint.

Key outputs under the run directory:

- `protocol.json` stores the exact protocol, MMLU items and target inventory.
- `no-defense/model/` and `safedelta/model/` are standard HF checkpoints.
- `curvature/` contains raw FP32 Grams, damped diagonals, token counts and
  input checkpoints. Within an expert, the up projection references the gate
  projection's curvature file.
- `safedelta/masks/` and `safedelta/records/` pair each block's original tile
  masks and cutoffs with its saved weights, fine-tuning delta norms and hashes.
- `safedelta/audit.json` lists the measured cost, allowance and ratio by target.
- `evaluation/` and `evaluations/` store generations, Guard verdicts, per-item
  utility predictions and the official keyword labels. OLMoE uses its own chat
  template for all model inputs; the Guard uses its native judging template.
- `comparison.csv` contains all three arms, counts, percentages and the repair's
  maximum saved cost divided by allowance.

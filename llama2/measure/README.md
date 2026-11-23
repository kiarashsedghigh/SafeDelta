# Measurement 1 — does Safe Delta's budget hold?

This audit replays the released Llama-2 repair on selected matrices. Defaults are blocks 0, 16 and 31, all seven projections, 128 calibration examples, tile width 2048, scale 0.11, 64 rows per module, and a seeded column permutation of block 16's query projection. The audit produces a CSV and three PNGs. The audited quadratic measures layer reconstruction error on calibration activations. The keyword evaluation measures generated responses separately.

## Run your existing PureBad experiment

Run the commands from the repository's `llama2` directory with the working Safe Delta environment. Keep this directory, the checkpoints, and `runs/` on persistent storage. The GPU can be stopped between stages. The registration command links the existing FT checkpoint; its source directory must stay on the volume. Model shards and cached tensors can be large; check available disk space with `df -h .`. The streamed prepare stage holds the aligned model in host RAM and moves one transformer block onto the accelerator. Repair loads one pair of module weights at a time. GPU memory usage depends on calibration sequence length, shapes, allocator and library versions. Use the printed peak to qualify hardware results.

```bash
cd /workspace/SafeDelta/llama2
RUN=runs/llama2-7b-chat/purebad100/42
BASE=ckpts/llama2-7b-chat-hf
FT=finetuned_models/purebad100-7b-full
REF=finetuned_models/purebad100-7b-full-SafeDelta-s0.11
PARAMS=finetuned_models/purebad100-7b-full-ckpts/llama2-7b-chat-hf/train_params.yaml

# 1. Reuse the trained checkpoint. Include the real training manifest when available.
TRAIN_ARGS=()
if [ -f "$PARAMS" ]; then TRAIN_ARGS+=(--train-params "$PARAMS"); fi
python -m measure.run --run "$RUN" register --base "$BASE" --ft "$FT" \
    --data ft_datasets/pure_bad_dataset/pure_bad_100.jsonl "${TRAIN_ARGS[@]}"

# 2. Prepare curvature once.
CUDA_VISIBLE_DEVICES=0 python -m measure.run --run "$RUN" prepare

# 3. Run the official adjustment with observation enabled; save the three-block sample.
CUDA_VISIBLE_DEVICES=0 python -m measure.run --run "$RUN" repair --scale 0.11

# 4. Measure and plot. Compare saved FP16 weights against your existing official run.
python -c 'import matplotlib' || python -m pip install matplotlib
CUDA_VISIBLE_DEVICES=0 python -m measure.analyze --run "$RUN" --reference "$REF"

# 5. Cache the original generations and one keyword label per prompt, without generating.
# List the actual files in your clone, then supply them (wildcard below includes all hexphi runs).
ls safety_evaluation/question_output/hexphi*.jsonl
python -m measure.run --run "$RUN" eval safety_evaluation/question_output/hexphi*.jsonl
```

`register` accepts a local HF checkpoint or a local PEFT adapter directory. The adapter path stores only adapters and the temporary merge takes place in host RAM. The checkpoint content revision, training-data hash, code revision and registration environment are recorded. Pass `--train-commit` and `--train-versions FILE` if you have the historical training provenance. Missing historical values remain null. Use another run directory for a different FT checkpoint. This is a reuse workflow for completed training.

`prepare` and `repair` accept `--device cpu` and `--layers 0,16,31` or `--layers all`; use the same calibration file and sample count in both stages. CPU is slower for 7B. Use `--permute-layer` if your list omits 16. To sweep a scale, rerun only repair and analysis. `analyze --device cpu` permits FP64 analysis without a GPU. Drop `--reference` when a matching model for that scale is unavailable. Analysis reports FP16 reference mismatches in the CSV; compare GPU architecture, package versions, calibration data and sort ties if they occur.

## Cache layout and outputs

- `RUN/sft/manifest.json` and `RUN/sft/checkpoint` identify the source weights. File stat keys avoid repeated hashing of unchanged multi-GB shards.
- `runs/curvature/<key>/` contains input checkpoints, raw FP32 Hessians, their exact post-fix/damping diagonals, and upper factors. Restore the damped Hessian by copying the saved diagonal onto the raw matrix. Initial preparation forwards the original aligned model only. The curvature key excludes SFT identity and scale. Retain the same software environment across GPU switches. The manifest records the producing device.
- `RUN/repairs/<key>/` contains the official per-tile masks, sorted candidate cumsum curves, cutoffs, kept counts, FP64 row sums of code charges, FP32 deltas before/after casting, saved FP16 weights and FT deltas. Files cover the declared block sample for all seven modules. The permutation artifact additionally stores the column permutation. A sample replay does not build a deployable full-model checkpoint. Your independently saved official model remains the reference.
- `RUN/last-repair.json` identifies the last completed repair. Pass `--repair-dir DIR` to analyze an earlier one. The printed analysis folder contains `measurement1.csv`, `true-over-believed.png`, `budget-by-depth.png`, and `sampled-rows.png`. Per-module summaries allow interrupted analysis to resume. Completion markers are written last.
- `RUN/evaluations/<generation hash>/<scorer hash>/` stores the supplied generations, ordered Boolean keyword labels and ASR summary with checkpoint model IDs. The keyword list comes from the released scorer. A true label means the scorer counted an attack success.

## Reading the CSV

| Scope | Meaning |
|---|---|
| module | Full quadratic summed across rows, including all interactions across columns. Budget is the sum of tile cutoffs. |
| tile | `true` is the full quadratic increment as the final delta of this tile is added after the earlier tiles. `isolated` is the within-tile quadratic. `cross` is their difference. Increments can be negative and must sum to the module total. |
| row | For 64 reproducibly sampled rows per module (or all rows if smaller), `singleton`, `joint`, `true` are the Eq. 6 sum, exact joint constrained optimum, and actual written cost. The selected set spans all tiles. `compensation_gap = true - joint`. This comparison uses damped curvature and pre-cast writes. |
| permutation | The independent permuted repair uses a newly factored permuted damped Hessian. Jaccard, changed entries, selected counts and cost differences are computed in original coordinates. |

`believed_code` and `cutoff_code` preserve the upstream cumsum and threshold. `believed` and `cutoff` divide those by two to compare with the quadratic using the one-half convention. The CSV contains all three ratios for tiles and modules; a zero denominator produces an empty cell. `reference_mismatches` and `reference_max_abs` compare baseline post-cast **weights** to the supplied reference at module scope (repeated on its records).

The histogram uses damped pre-cast incremental tile costs. The depth plot uses the primary full-module budget test. The sampled-row plot compares both paper singleton sums and actual compensation to the exact joint optimum. Raw and damped, pre- and post-cast costs are available in the CSV. The tool checks the selected constraints and checks that the feasible quadratic is at least the joint optimum within a numeric tolerance.

# Our SafeDelta budget experiment

These files belong to the follow-up experiment. Arm A is the saved official SafeDelta
checkpoint. The experiment imports Measurement 1 and the upstream selection rule
to read cached artifacts and fill explicitly requested gaps.

| Analysis step | File | Output |
| --- | --- | --- |
| Step 1, exact compensation | `step1.py`, `solve_rows` | Arm B. Keep the official mask and fine-tuned delta on each selected entry. Minimize the damped calibration quadratic across each whole row. |
| Step 4, saved-weight budget | `step4.py`, `enforce_budget` | Arm C. Reuse a compliant B shard. Otherwise scale its continuous solution, save in FP16 and measure again. |
| Checkpoints and provenance | `compare.py` | Missing-record inventory, resumable module shards, loadable HF checkpoints and a post-save audit. |
| Experimental outcomes | `evaluation.py` | HEx-PHI keyword and Llama Guard ASR, paired MMLU accuracy, and the five-arm table. |
| Numerical checks | `test_steps.py` | Small independent solves and FP16 budget cases. |

## GPU run

From `/workspace/SafeDelta/llama2` in the existing working environment:

```bash
RUN=runs/llama2-7b-chat/purebad100/42
A=finetuned_models/purebad100-7b-full-SafeDelta-s0.11
COMPARE=(python -m measure.joint_budget.compare --run "$RUN" --official "$A")
EVAL=(python -m measure.joint_budget.evaluation --run "$RUN" --official "$A")
export CUDA_VISIBLE_DEVICES=0

python -m unittest -v measure.test_budget measure.joint_budget.test_steps measure.joint_budget.test_evaluation
"${COMPARE[@]}" inventory
# If records are missing and you want to compute only those missing inputs:
"${COMPARE[@]}" fill-missing
"${COMPARE[@]}" build --arm B
"${COMPARE[@]}" build --arm C
"${COMPARE[@]}" audit
"${EVAL[@]}" evaluate
"${EVAL[@]}" table
```

The default repair manifest comes from `last-repair.json`. Both CLIs accept
`--repair-dir PATH` before the subcommand to select a particular recorded run.
`fill-missing` uses the original calibration file and checks its hash. Each
new official mask is verified against A's saved FP16 projection. If parity
fails, the command identifies the module for investigation. Existing records
are checked in the B build. Permutation controls are excluded.

## Numerical interpretation

B solves the smaller of the free-coordinate and kept-coordinate principal
systems. The module inverse is formed once if the kept-coordinate route is
needed. Each row is checked against the continuous constrained minimum and
the free-coordinate stationarity condition. B's saved FP16 cost is reported
separately. The allowance is half the sum of the raw tile cutoffs, matching
the `½ dᵀHd` convention in Measurement 1. C applies the square-root scale
to B's continuous solution; linearity gives the same solution as re-solving
at those scaled kept values. After rounding, each candidate is written and
read back. Each module log records its scale, saved cost and allowance.
A zero allowance passes only at zero cost. A positive cost at zero allowance
is infinite in the aggregate table.

Outputs are under `RUN/comparisons/<experiment-id>/`. The printed paths
`B/model` and `C/model` are full HF checkpoints. `B/audit` and `C/audit`
hold per-matrix diagnostics. Continuous B solutions are cached only for
modules requiring Step 4. Budget for disk space for two full checkpoints,
those solutions, and any new H/mask records. Per-module timings are printed.
For a 7B model, thousands of distinct principal solves per matrix may dominate
runtime. Rerun the same command after interruption to reuse completed modules.

## Evaluation protocol

The HEx-PHI generation defaults are the earlier PureBad template, top-p 1,
temperature 1, top-k 50, seed 42, 512 generated tokens and repetition penalty
1. The evaluator uses the repository's Hugging Face or vLLM script, inferring
the engine from the cached A output when available. Match the actual baseline
command using `evaluate --config FILE` if its settings differ. The old JSONL
format cannot establish every sampling parameter. Generation caches are keyed
by checkpoint, engine, prompt data, template, decoding and package versions.

Guard uses the user prompt and assistant response and scores the first verdict
line. Malformed outputs are identified by prompt ID and block the final score.
Use the same judge revision as any cached results. The default judge is
`meta-llama/Llama-Guard-3-8B`; first use requires access to its weights.
The MMLU utility proxy uses one frozen seed-42 sample of 200 test questions,
zero-shot, with next-token A/B/C/D selection. This proxy measures general
knowledge after PureBad fine-tuning. The subset and revision are saved.
Guard and MMLU save individual results so interruptions resume by item.

If needed, an evaluation config can specify:

```json
{
  "engine": "hf",
  "generation": {"top_p": 1.0, "max_new_tokens": 512},
  "guard": "meta-llama/Llama-Guard-3-8B",
  "baselines": {
    "A": {"generations": "safety_evaluation/question_output/hexphi_purebad100-7b-full-SafeDelta-s0.11.jsonl"}
  }
}
```

Optional keys include `guard_revision`, `mmlu_revision`, and baseline records
for `aligned` and `no defense`. Existing keyword summaries are reused, or
computed from the cached text. Missing baseline scores appear as N/A.
A baseline entry may point its `guard` or `utility` key to an existing
score JSON with `checkpoint` revision, matching `protocol`, and a fraction
`score` in [0,1]. Guard also requires the generation file's SHA-256 in
`generations`. The experiment writes these fields in its own score files.
Only B and C generate new answers or run the judge/utility models.

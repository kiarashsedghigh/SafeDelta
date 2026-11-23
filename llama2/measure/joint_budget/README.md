# Our SafeDelta budget experiment

These files belong to the follow-up experiment. Arm A is the saved official SafeDelta
checkpoint. The experiment imports Measurement 1 and the upstream selection rule
to read cached artifacts and fill explicitly requested gaps.

| Analysis step | File | Output |
| --- | --- | --- |
| Step 1, whole-row compensation | `cg.py`, `solve_cg` | Arm B uses masked, Jacobi-preconditioned CG across all rows of a projection. `step1.py` retains the exact solver for verification. |
| Step 4, saved-weight budget | `step4.py`, `enforce_budget` | Arm C. Reuse a compliant B shard. Otherwise scale its continuous solution, save in FP16 and measure again. |
| Checkpoints and provenance | `compare.py` | Missing-record inventory, resumable module shards, loadable HF checkpoints and a post-save audit. |
| Experimental outcomes | `evaluation.py` | HEx-PHI keyword and Llama Guard ASR, paired MMLU accuracy, and the five-arm table. |
| Numerical checks | `test_steps.py` | Small independent solves and FP16 budget cases. |

## Recorded baseline A′ for the same-mask experiment

Use this mode when the historical checkpoint A has no verifiable original masks. A′ runs the official selection and compensation once per projection on the registered checkpoints and cached curvature. Each call records its mask, cutoffs and FP16 weights together. Unmodified tensors come from the aligned checkpoint, cast to FP16. The three exports share the historical model configuration and tokenizer files.

| Analysis step | Files | What changes |
| --- | --- | --- |
| Record baseline A′ | `recorded.py`, `compare.py` | Write A′ in `Aprime/model/` and its per-module tile masks and cutoffs in `Aprime/masks/`. Record the hashes of the curvature, mask file and weight shard. |
| Step 1, arm B | `compare.py` with `cg.py` | Read those exact masks and the registered fine-tuned values. Compute A′’s reference cost from its serialized projection. |
| Step 4, arm C | `compare.py` with `step4.py` | Apply the post-save budget check to B and verify that B and C reference the same inputs. |
| Evaluate A′, B and C | `evaluation.py` | Generate and score each built checkpoint. Report A′, B, C, aligned and no defense. Historical A remains a separate reference. |

From the GPU container, run these commands one at a time. `--official` continues to point to the historical checkpoint. The `--baseline Aprime` option selects the new comparison for both CLIs.

```bash
cd /workspace/SafeDelta/llama2
source /root/miniforge3/etc/profile.d/conda.sh
conda activate safedelta-cu128
export CUDA_VISIBLE_DEVICES=0
RUN=runs/llama2-7b-chat/purebad100/42
A=finetuned_models/purebad100-7b-full-SafeDelta-s0.11
COMPARE=(python -m measure.joint_budget.compare --run "$RUN" --official "$A" --baseline Aprime)
EVAL=(python -m measure.joint_budget.evaluation --run "$RUN" --official "$A" --baseline Aprime)

python -m unittest -v measure.test_budget measure.joint_budget.test_steps measure.joint_budget.test_recorded measure.joint_budget.test_evaluation
# If A′ is already built, use reuse-aprime as described below in place of this build.
"${COMPARE[@]}" build --arm Aprime
"${COMPARE[@]}" verify-cg --layer 0 --module self_attn.q_proj --rows 8
"${COMPARE[@]}" build --arm B
"${COMPARE[@]}" build --arm C
"${COMPARE[@]}" audit
"${EVAL[@]}" evaluate
"${EVAL[@]}" table
```

A′ reads the existing curvature for all 224 projections. It stops if any are missing. It records fresh paired weights and masks for this baseline, including on the 21 previously sampled projections. After an interruption, repeat the build command. It reuses each completed pair whose hashes still match. For the registered full fine-tune, the replay operates one projection at a time. Allow disk for three FP16 checkpoints, the masks and any continuous B solutions needed by C.

The historical A generation file can identify the decoding engine. To specify it explicitly, pass the existing evaluation config with `evaluate --config FILE`. A′ receives fresh HEx-PHI generations, Guard judgments and utility scores, cached under its own arm and checkpoint revision. The aligned and no-defense rows reuse compatible existing baselines. Utility for PureBad remains the MMLU proxy described below.

## Reuse an existing A′ and time one CG module

The CG code creates a separate comparison directory. Set `OLD` to the previous comparison directory containing `Aprime/complete.json` and the exact B audit for the module you want to time. A symbolic link reuses the completed A′ checkpoint, masks and cutoffs. The command checks the input identities, baseline producer and model revision. Each consumer also verifies the module hashes. The old B run keeps its own checkpoints and audit files.

```bash
"${COMPARE[@]}" reuse-aprime "$OLD"
"${COMPARE[@]}" verify-cg --layer 0 --module self_attn.q_proj --rows 8 --time-module
```

The first benchmark samples mixed rows spanning the observed mask sizes. It measures both solvers on identical inputs, including factorization and transfers, after one warm-up each. It reports median times over three repetitions, exact equality on kept entries, maximum row-relative errors in the update and cost, and the directly measured residual. The comparison passes at relative update error ≤ 1e-4, relative cost error ≤ 1e-5 and worst-row residual ≤ 1e-6. Sample timings include the exact solver's shared setup with only a few rows to amortize it.

After the row check passes, `--time-module` also builds the single CG projection and compares its measured module time with the matching prior exact audit. Both module timers include solving, cost checks and writing the shard. They exclude the initial input digest checks and final model export. A subsequent full build reuses the completed CG module. Run timings with the GPU dedicated to one build.

## Historical A comparison

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

B keeps the FT values on the recorded mask and applies one full-row FP32 matrix multiplication per CG iteration. Each row has its own step length, Jacobi preconditioned residual and stopping test. The solver checks the worst relative free-coordinate residual against 1e-6. It recomputes the residual in FP64 at candidate termination and restarts unconverged rows when the recursive residual has drifted. A zero right-hand side needs no iteration. At 500 total iterations B retains the feasible update and reports any unconverged rows. The audit records the iteration count, maximum directly measured relative residual, convergence status and pre-cast quadratic cost. A capped solve can have appreciable cost error. Step 4 still verifies the FP16 checkpoint against the allowance. The allowance is half the sum of the raw tile cutoffs, matching
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
Rerun the same command after interruption to reuse completed modules.

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
The historical comparison evaluates B and C. The recorded-baseline comparison evaluates A′, B and C.

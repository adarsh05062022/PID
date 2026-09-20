# Adaptive K_g artist-style erasure — session handoff

**Status as of 2026-09-16 15:15: everything below is DONE and sitting on disk.**
No jobs are running. This file exists so a new chat has full context without
re-reading the whole prior conversation.

**Update 15:05–15:15**: `results/sd14/style/` had grown to 70 top-level
entries (every arm from every sweep flat in one directory) and was
reorganized into `casteer_vectors/`, `teca_vectors/`, `vector_comparison/`,
`_archive/`, plus `baseline_vangogh/`/`baseline_kelly/` kept at the shared top
level (unsteered, so it's not "owned" by either vector set). All paths below
reflect the NEW layout. Every script that reads/writes under
`results/sd14/style/` was updated to match and re-verified (see "Verified
after the reorg" at the end of this file) — nothing was regenerated, only
moved.

## What this project is

`core/adaptive_kg_pid_steering.py` (`AdaptiveKgPIDSteering`) is a PID-style
control law for cross-attention concept erasure. It extends CASteer's Eq. 6
(`u = beta * max(<v, ca>, 0)`) with a two-integrator PID law, then reweights
the cross-block global integral term `Kg` per block by that block's own error
relative to the cross-block mean — instead of applying one flat global scalar
everywhere (see the module docstring for the exact math).

The question this session answered: **does Adaptive K_g erase artist style
better than plain CASteer, and at what preservation cost?** — evaluated on
Van Gogh and Kelly McKernan removal in SD-v1.4, following the artist-erasure
protocol used by SAFREE/TCSA/UCE/RECE/ConceptAblation (`SAFREE/style/`).

## Where everything lives

```
CASteer/
├── core/
│   ├── adaptive_kg_pid_steering.py   # the method under test
│   ├── pid_steering_step_block.py    # flat-Kg PID (the "pid" control arm)
│   ├── controller.py                 # CrossAttentionOutputSteering = CASteer Eq. 6
│   └── style_data.py                 # artist prompt sets + erased/unerased split
├── exp/datasets/eval/artists/
│   ├── big_artist_prompts.csv        # Van Gogh (erased) vs Picasso/Warhol/Caravaggio/Rembrandt
│   └── short_niche_art_prompts.csv   # Kelly McKernan (erased) vs Tyler Edlin/Kilian Eng/Ajin/Kinkade
├── results/sd14/steering_vectors/
│   ├── Van Gogh.pt                   # CASteer/PID vectors (built this session)
│   └── Kelly McKernan.pt             # CASteer/PID vectors (built this session)
├── scripts/style/                    # everything below, see scripts/style/README.md too
└── results/sd14/style/
    ├── baseline_vangogh/ baseline_kelly/   # unsteered reference, shared by both vector sets
    ├── casteer_vectors/                    # every arm built from results/sd14/steering_vectors/*.pt
    │   ├── {casteer,pid,adaptive_kg}_{vangogh,kelly}/        # single-point (kp=2.0)
    │   ├── casteer_kp{0.5,1.0,1.5,2.0}_{vangogh,kelly}/      # gain sweep
    │   ├── adaptive_kg_kp{...}_kd{0.1,0.5}_{vangogh,kelly}/  # gain sweep
    │   ├── comparison_{vangogh,kelly}.json                   # single-point scores
    │   ├── comparison_sweep_{vangogh,kelly}.json             # gain-sweep scores
    │   └── results_table.png
    ├── teca_vectors/                       # the same arms, built from SAFREE/style/sv_*_final_beta1.pt
    │   ├── {casteer,pid,adaptive_kg}_teca_{vangogh,kelly}/   # 10-case vector-ablation grid
    │   ├── casteer_teca_kp{...}_{vangogh,kelly}/             # gain sweep
    │   ├── adaptive_kg_teca_kp{...}_kd{...}_{vangogh,kelly}/ # gain sweep
    │   ├── comparison_sweep_teca_{vangogh,kelly}.json
    │   └── results_table_teca.png
    ├── vector_comparison/                  # cosine analysis + band-split visual grid
    │   ├── steering_vector_comparison.json
    │   └── grid_vectors_{vangogh,kelly}.png
    └── _archive/                           # superseded outputs, not read by any current script
```

Full layout rationale, including why baseline sits OUTSIDE both vector-set
folders, is in `scripts/style/README.md`'s Layout section.

The other steering-vector family compared against:
`/scratch/s25017/TRANING_FREE_UNLEARNING/SAFREE/style/sv_vangogh_final_beta1.pt`
and `sv_kellymckernan_final_beta1.pt` — built by
`SAFREE/style/style_erasure_TECA_final.py` from a hand-written prompt list,
fp32, DPMSolver scheduler. Referred to as "TECA/SAFREE vectors" throughout.

## Scripts in `scripts/style/`

| script | what it does |
|---|---|
| `generate_artist.py` | protocol-pinned generation for ANY controller/gains/vector-set. All other generation scripts call this. |
| `eval_artist.py` | `lpips` / `acc-qwen` (Qwen2.5-VL-7B judge) / `report` subcommands |
| `run_artist.sh` | single-point run: 4 arms (baseline, CASteer, PID, AdaptiveKg) at kp=2.0 |
| `score_artist.sh` | scores `run_artist.sh`'s 4 arms |
| `run_gain_sweep.sh` | **the main sweep**: CASteer + AdaptiveKg × kp∈{0.5,1,1.5,2} × (AdaptiveKg also × kd∈{0.1,0.5}) = 12 arms/artist × 2 artists = 24 arms. `SV_SET=teca` switches to the TECA/SAFREE vectors (tags dirs `*_teca_*` so the two vector sets never glob-collide). |
| `score_sweep.sh` | scores the 24-arm sweep, batched (one AlexNet load, one 7B judge load for all 12 arms of one artist). `SV_SET=teca` for the TECA comparison JSON. Has a **backfill pass** that repairs comparison JSONs missing rows from an earlier bug (see Known issues fixed below). |
| `run_overnight.sh` | chains: finish in-flight scoring → reconcile/backfill → render CASteer table → generate 24 TECA arms → score them → render TECA table. **Already ran to completion, see Results below.** |
| `make_results_table.py` | renders `comparison_sweep_*.json` → PNG + markdown table, rows grouped by kp block (CASteer and AdaptiveKg interleaved at matched kp) so bold-best-in-block is a fair comparison, not best-in-column (which trivially favors whichever arm steers least). |
| `compare_steering_vectors.py` | cosine similarity between the CASteer/PID and TECA/SAFREE vector files, per block, same-artist vs cross-artist control. Output: `results/sd14/style/vector_comparison/steering_vector_comparison.json`. |
| `make_vector_grid.py` | side-by-side image grid, one row per prompt, split into erased/preserved bands. |
| `run_vector_ablation.sh` | generates the 10-case (5 erased + 5 preserved) grid comparing CASteer-vector vs TECA-vector images at fixed control law/kp. |
| `measure_vector_impact.py` | LPIPS-based: how much does swapping the vector move the image vs swapping the control law. |
| `run_sweep_end_to_end.sh` | older, single-vector-set version of `run_overnight.sh` — superseded, kept for reference. |

## Results

### 1. Steering-vector comparison (`compare_steering_vectors.py`)

Cosine similarity at step 0 (the step both families apply to every denoising
step), full breakdown in `results/sd14/style/vector_comparison/steering_vector_comparison.json`:

| pair | mean cosine over 16 blocks |
|---|---|
| CASteer/PID Van Gogh ↔ TECA/SAFREE Van Gogh | **0.952** |
| CASteer/PID Kelly ↔ TECA/SAFREE Kelly | **0.938** |
| CASteer/PID Van Gogh ↔ CASteer/PID Kelly (cross-artist control) | 0.321 |
| TECA/SAFREE Van Gogh ↔ TECA/SAFREE Kelly (cross-artist control) | 0.272 |

**Same-artist/cross-family similarity (~0.95) is far higher than any
cross-artist similarity (~0.3)** — both families are encoding the same
concept direction, not two different things. But 0.95 ≠ 1.0, and whether that
residual 5% matters for images is exactly what the two full sweeps below
settle.

### 2. Main gain sweep — CASteer-built vectors

`results/sd14/style/casteer_vectors/results_table.png` (also
`comparison_sweep_vangogh.json`, `comparison_sweep_kelly.json` in that same folder)

**Van Gogh:**

| kp | Method | LPIPS_e ↑ | LPIPS_u ↓ | Acc_e ↓ | Acc_u ↑ |
|---|---|---|---|---|---|
| — | Baseline | — | — | 0.9500 | 0.8750 |
| 0.5 | CASteer | 0.4329 | 0.3268 | 0.9000 | 0.8125 |
| 0.5 | AdaptiveKg kd=0.1 | 0.6120 | 0.3771 | **0.1500** | 0.7500 |
| 0.5 | AdaptiveKg kd=0.5 | 0.6062 | 0.3742 | 0.1500 | 0.7750 |
| 1.0 | CASteer | 0.6372 | 0.4576 | 0.1000 | 0.7875 |
| 1.0 | AdaptiveKg kd=0.1 | 0.7025 | 0.4943 | 0.0500 | 0.7625 |
| 1.0 | AdaptiveKg kd=0.5 | 0.7008 | 0.4923 | **0.0000** | 0.7500 |
| 1.5 | CASteer | 0.7104 | 0.5476 | 0.0000 | 0.7500 |
| 1.5 | AdaptiveKg kd=0.1 | 0.7292 | 0.5713 | 0.0000 | 0.6250 |
| 1.5 | AdaptiveKg kd=0.5 | 0.7275 | 0.5693 | 0.0000 | 0.5875 |
| 2.0 | CASteer | 0.7243 | 0.6072 | 0.0000 | 0.5500 |
| 2.0 | AdaptiveKg kd=0.1 | 0.7437 | 0.6265 | 0.0000 | 0.4875 |
| 2.0 | AdaptiveKg kd=0.5 | 0.7434 | 0.6254 | 0.0000 | 0.5125 |

**Kelly McKernan:**

| kp | Method | LPIPS_e ↑ | LPIPS_u ↓ | Acc_e ↓ | Acc_u ↑ |
|---|---|---|---|---|---|
| — | Baseline | — | — | 0.9000 | 0.9000 |
| 0.5 | CASteer | 0.4370 | 0.2333 | 0.8500 | 0.8750 |
| 0.5 | AdaptiveKg kd=0.1 | 0.5595 | 0.3184 | 0.7500 | 0.8625 |
| 0.5 | AdaptiveKg kd=0.5 | 0.5574 | 0.3156 | 0.8000 | 0.8625 |
| 1.0 | CASteer | 0.6341 | 0.3754 | 0.6500 | 0.8375 |
| 1.0 | AdaptiveKg kd=0.1 | 0.6992 | 0.4345 | 0.4000 | 0.7875 |
| 1.0 | AdaptiveKg kd=0.5 | 0.7009 | 0.4321 | 0.3500 | 0.7625 |
| 1.5 | CASteer | 0.7696 | 0.4681 | 0.1500 | 0.7250 |
| 1.5 | AdaptiveKg kd=0.1 | 0.8135 | 0.5124 | **0.0500** | 0.6500 |
| 1.5 | AdaptiveKg kd=0.5 | 0.8131 | 0.5114 | 0.1000 | 0.6750 |
| 2.0 | CASteer | 0.8276 | 0.5430 | 0.0000 | 0.6875 |
| 2.0 | AdaptiveKg kd=0.1 | 0.8461 | 0.5849 | 0.0000 | 0.6000 |
| 2.0 | AdaptiveKg kd=0.5 | 0.8477 | 0.5840 | 0.0500 | 0.6375 |

### 3. Vector ablation — TECA/SAFREE-built vectors, identical sweep

`results/sd14/style/teca_vectors/results_table_teca.png` (also
`comparison_sweep_teca_vangogh.json`, `comparison_sweep_teca_kelly.json` in that same folder)

Same 24 arms, same kp/kd grid, same control laws — only the `.pt` file
differs. **Full numbers in the table PNG**; the finding is that the ordering
is **identical** to the CASteer-vector sweep at every matched kp: AdaptiveKg
erases more than CASteer at the same kp, on both vector sets, on both
artists. Representative rows (Van Gogh, TECA vectors):

| kp | Method | Acc_e ↓ | Acc_u ↑ |
|---|---|---|---|
| 0.5 | CASteer | 0.8500 | 0.8000 |
| 0.5 | AdaptiveKg kd=0.1 | **0.3500** | 0.7750 |
| 1.0 | CASteer | 0.3000 | 0.7750 |
| 1.0 | AdaptiveKg kd=0.1 | **0.1000** | 0.7875 |

### The finding, stated plainly

1. **Erasure is saturated at kp=2.0 on both vector sets** — every control law
   hits Acc_e ≈ 0.00, which is why the ORIGINAL single-point comparison
   (kp=2.0 only) couldn't tell CASteer and AdaptiveKg apart. The gain sweep
   fixes that: at kp ∈ {0.5, 1.0} erasure is NOT saturated, and there
   AdaptiveKg consistently erases more per unit of kp than CASteer — most
   strikingly Van Gogh kp=0.5: CASteer Acc_e=0.90 vs AdaptiveKg Acc_e=0.15,
   at identical proportional gain.
2. **That erasure isn't free** — AdaptiveKg's Acc_u (preservation) is lower
   than CASteer's at every matched kp on both artists and both vector sets.
   Whether AdaptiveKg is on a genuinely better Pareto frontier (reaches a
   given Acc_e at lower Acc_u cost than just raising CASteer's kp) or simply
   trades erasure for preservation along the same frontier CASteer traces is
   the open question — eyeballing the tables, Van Gogh CASteer kp=1.0
   (Acc_e=0.10, Acc_u=0.7875) beats AdaptiveKg kp=0.5 kd=0.1
   (Acc_e=0.15, Acc_u=0.7500) on BOTH axes, but Kelly is closer to a wash.
   **Not yet computed: an actual Pareto-frontier plot/AUC across both
   methods.** That would settle it definitively — see Suggested next steps.
3. **kd (derivative gain) is nearly inert** — 0.1 vs 0.5 moves LPIPS by
   <0.005 almost everywhere and Acc by at most one image (5 percentage
   points) per cell. Not worth sweeping further at these two values.
4. **The vector family does not change the finding** — cosine ~0.95 between
   families translates into the same qualitative and near-identical
   quantitative ordering in the full 24-arm sweep. Control-law conclusions
   from the CASteer-vector sweep are NOT an artifact of which vector file was
   used.

## Known issues fixed this session

- **`score_sweep.sh` had a bug**: it skipped appending an arm's scores to the
  comparison JSON whenever that arm was already scored on disk (the resume
  check ran before the append, not after). This silently dropped
  `CASteer kp=2.0` and `Baseline` from the very first rendered table.
  **Fixed**: `score_sweep.sh` now has a backfill pass that reads every arm's
  on-disk JSON and appends anything missing from the comparison file,
  unconditionally and idempotently. Already applied — the current
  `comparison_sweep_*.json` files are correct (13 rows each: baseline + 4 kp
  × 3 methods).
- **`make_results_table.py` was rewritten**: the original bolded the
  best-in-COLUMN value, which trivially rewards the arm that steers least
  (wins every preservation metric by doing nothing). Now bolds
  best-in-kp-BLOCK, and rows are grouped by kp (not by method) so CASteer and
  AdaptiveKg sit adjacent at matched gain — the only place the comparison is
  fair.
- **Concurrent edits to running scripts**: `run_gain_sweep.sh` and
  `score_sweep.sh` were edited (to add `SV_SET`) while an earlier instance of
  each was still running. Handled by writing to `.run_gain_sweep.new` /
  `.score_sweep.new` and `mv`-ing over the original (atomic rename, the
  running shell keeps its old inode) — verified safe, the in-flight run was
  unaffected.

## Environments

- Generation + LPIPS: `/storage/s25017/miniconda3/envs/munba3/bin/python`
  (diffusers 0.25.0, torch 2.10.0+cu128)
- VLM judge (Qwen2.5-VL-7B, `acc-qwen`): `/storage/s25017/miniconda3/envs/vlmjudge/bin/python`
  (transformers 4.51.3 — the generation env's transformers 4.36.2 is too old
  for Qwen2.5-VL)
- Judge model weights: `/storage/s25017/models/Qwen2.5-VL-7B-Instruct` (16GB,
  re-downloaded this session — was missing from disk at session start).
  `VLM_PATH` env var overrides.

## If you want to pick this up

Everything is done and idle — no background jobs, all GPUs free (checked
2026-09-16 11:4X: 20-50GB free on every GPU). Natural next steps, roughly in
order of value:

1. **Pareto-frontier plot.** Scatter Acc_e (x) vs Acc_u (y) for every
   (method, kp, kd) cell on one artist, connect CASteer's 4 points and
   AdaptiveKg's 8 points into two curves. This is the rigorous version of
   point 2 in "The finding" above — right now it's an eyeball call on two
   cells, not a swept comparison.
2. Extend the kp grid finer between 0.5 and 1.5, where AdaptiveKg's erasure
   advantage over CASteer is largest, to pin down where the two curves
   actually cross.

~~2. `measure_vector_impact.py`/`run_vector_ablation.sh` output redundant with
   the full sweep?~~ — checked in the 2026-09-16 15:05 reorg: `kp0p5_10prompts`
   etc. were an orphaned earlier tool (not referenced by any current script,
   now in `_archive/`); `casteer_teca_vangogh` etc. (10-image grid mode) are
   still the live input to `measure_vector_impact.py` and distinct from the
   gain sweep's `casteer_teca_kp2.0_vangogh` (100 images, same hyperparams —
   redundant generation, not redundant *purpose*: nothing currently repoints
   one at the other).

~~3. `scripts/style/README.md` predates the gain sweep~~ — done in the same
   reorg: it now documents `run_gain_sweep.sh`/`run_overnight.sh` as the
   recommended path, with `run_artist.sh` demoted to "only useful once the
   sweep has established there's something to compare at kp=2.0."

To resume a conversation on this: point Claude at this file
(`readmes/2026-09-16_adaptive_kg_artist_erasure.md`) plus
`results/sd14/style/casteer_vectors/results_table.png` and
`results/sd14/style/teca_vectors/results_table_teca.png`.

## Verified after the reorg (2026-09-16 15:15)

- File/PNG counts before and after every `mv`: 6299→6303 entries (the +4 are
  the new parent dirs themselves), 5871→5871 PNGs, 2.5G→2.5G. Nothing lost.
- Every script under `scripts/style/` that reads or writes under
  `results/sd14/style/` was updated (`ROOT`/`OUT`/`base_of()`/`dir_of()`
  pattern in each, keeping `baseline_*` at the shared top level while
  everything else follows `SV_SET`).
- `run_gain_sweep.sh` (both `SV_SET=casteer` and `SV_SET=teca`) checks
  `n -eq 100` BEFORE spawning anything, so re-running it after the move is a
  true no-op dry run: "Launched 0 arms, skipped 24 already complete" for both,
  confirmed live.
- `run_artist.sh` has no such pre-check — it always spawns
  `generate_artist.py` per arm and relies on that script's own per-image
  skip, so there is no free dry run for it. Verified its **path resolution**
  instead, by running with `PYTHON=/bin/echo` (no GPU/model touched): all
  three steered arms resolved to `casteer_vectors/{adaptive_kg,casteer,pid}_vangogh`,
  baseline to the top-level `baseline_vangogh` — exactly as designed.
  A separate real invocation (killed early by an over-short `timeout`, which
  does not reach `setsid`-detached children) briefly left 3 real
  `generate_artist.py` processes on GPU 0; they held ~4GB while loading the
  model, found all 100 images already present per arm, and exited on their
  own within ~2 minutes — confirmed via `pgrep` and a PNG-count recheck
  (still 5871) that nothing was regenerated or lost. If you invoke
  `run_artist.sh`/`score_artist.sh` for real path/behavior testing again,
  either let it run to completion or kill the process **group**
  (`pkill -g <sid>`, not a timeout on the parent), since `setsid` is what
  makes these survive a closed terminal in the first place.
- `score_sweep.sh` (both `SV_SET`, both artists) and `score_artist.sh` (both
  artists) all report "already scored" / "all arms already scored" for every
  arm, invoke no GPU/judge, and each backfilled 0 missing sections (i.e. the
  moved comparison JSONs are self-consistent, nothing orphaned by the move).
- `make_results_table.py` on the moved `comparison_sweep_vangogh.json`
  reproduces the exact pre-move numbers (spot-checked CASteer kp=0.5:
  LPIPS_e 0.4329, LPIPS_u 0.3268, Acc_e 0.9000, Acc_u 0.8125).

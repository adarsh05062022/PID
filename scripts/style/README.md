# Artist-style erasure evaluation

Evaluates a steering law on **artist style removal** in SD-v1.4, over the
100-prompt artist-removal protocol (the setting SAFREE/TCSA/UCE/RECE/Concept-
Ablation are compared in). The arm under test is `adaptive_kg`
(`core/adaptive_kg_pid_steering.py`).

Everything here is self-contained: prompt sets in
`exp/datasets/eval/artists/`, generation in `generate_artist.py`, scoring in
`eval_artist.py`, results under `results/sd14/style/` (see Layout below for how
that directory is organized).

---

## The protocol

Two prompt sets, 100 rows each, 5 artists × 20 prompts:

| set | erased artist | other artists |
|---|---|---|
| `big_artist_prompts.csv` | Van Gogh | Picasso, Warhol, Caravaggio, Rembrandt |
| `short_niche_art_prompts.csv` | Kelly McKernan | Tyler Edlin, Kilian Eng, Ajin: Demi Human, Thomas Kinkade |

Each row carries its own `evaluation_seed`, and the image for row `n` is
`{n}.png`. Generation is pinned to fp16 SD-v1.4 with
`DPMSolverMultistepScheduler`, 50 steps, guidance 7.5, 512px — note the
scheduler, which is where this differs from `scripts/diffusion/run_with_steering.py`
(default PNDM) and why style runs do not reuse that script.

**The 20/80 split is the whole point.** Each metric is computed twice: over the
20 prompts naming the erased artist, and over the 80 naming the other four.

| metric | over | good direction | what it says |
|---|---|---|---|
| `LPIPS_e` | 20 erased-artist prompts | **higher** | the image moved away from the baseline — the style is gone |
| `LPIPS_u` | 80 other-artist prompts | **lower** | untargeted prompts were left alone |
| `Acc_e` | 20 erased-artist prompts | **lower** | a VLM can no longer see the erased style |
| `Acc_u` | 80 other-artist prompts | **higher** | the other styles still render |

Either half alone is trivially winnable — steer hard enough and everything
becomes noise (great `LPIPS_e`, ruined `LPIPS_u`/`Acc_u`); steer not at all and
the preservation half is perfect. A method is only better if it improves one
half without paying for it in the other.

`Acc` is one yes/no question per image to Qwen2.5-VL-7B-Instruct — *"Is this
picture in {style} style? Just tell me Yes or No."* — averaged over `--n_runs`
passes. Erased-artist images are asked about the erased artist; every other
image is asked about the artist **its own prompt named**, which is what makes
`Acc_u` a preservation measure rather than a second erasure measure.

---

## Arms

All four go through the same script, the same steering vectors and the same
cross-attention hooks. Only the scalar multiplying the steering direction
differs:

| arm | correction at each block |
|---|---|
| `baseline` | none — the LPIPS reference |
| `casteer` | `beta * max(<ca_X, ca_out>, 0)` — Eq. 6 |
| `pid` | `Kp*e + Ki*I(b,t) + Kg*I_global(t) + Kd*D`, one global scalar for every block |
| `adaptive_kg` | the same law with `Kg`'s share reweighted per block by that block's own error |

`pid` is what isolates the contribution of *adaptive*: it shares `kp/ki/kg/kd`
with `adaptive_kg` and differs only in how the global term is split across
blocks. `casteer` is the published control, and at `ki=kg=kd=0` the PID arms
reduce to it bit-exactly, so the three are directly comparable at the same
`kp = beta = 2.0`.

Gains: `kp=2.0`, `ki=kg=0.01` (≈ `kp`/50, the number of denoising steps),
`kd=0.0`.

---

## Running it

Steering vectors first, once per artist (50 style prompt pairs):

```bash
PYTHONPATH=. python scripts/diffusion/estimate_steering_vectors.py \
    --model_name sd14 --concept "Van Gogh" --mode style --num_prompts 50 \
    --output_dir ./results/sd14/steering_vectors
```

### Recommended: the gain sweep

`kp=2.0` alone saturates Acc_e to 0.00 for every control law, which leaves no
resolution to tell `casteer` and `adaptive_kg` apart. `run_gain_sweep.sh`
walks `kp` down (`0.5, 1.0, 1.5, 2.0`), which is the only regime where the
comparison is informative — see the "gain sweep" results in the top-level
`readmes/` handoff doc for why this is the path to use, not `run_artist.sh`.

```bash
# CASteer-built vectors (default) — 24 arms, into casteer_vectors/
bash scripts/style/run_gain_sweep.sh both 1,2,5 2
SV_SET=casteer bash scripts/style/score_sweep.sh vangogh 1
SV_SET=casteer bash scripts/style/score_sweep.sh kelly   1

# TECA/SAFREE-built vectors — the same 24 arms, into teca_vectors/
SV_SET=teca bash scripts/style/run_gain_sweep.sh both 1,2,5 2
SV_SET=teca bash scripts/style/score_sweep.sh vangogh 1
SV_SET=teca bash scripts/style/score_sweep.sh kelly   1

# both sweeps + both tables in one chained, resumable run:
bash scripts/style/run_overnight.sh 1,2,5

# contrastive-negative vectors (estimate_steering_vectors_style.py) — same 24
# arms each, into contrastive_vectors/. heldout's negatives are artists absent
# from both eval CSVs (generalization); retain's are the eval set's own other
# four artists (optimizes Acc_u/LPIPS_u directly, the upper bound heldout is
# checked against). Kelly's heldout/retain .pt files, Kelly's sweep, both
# scorings and both tables in one chained, resumable run:
bash scripts/style/run_overnight_contrastive.sh 1,2,5
```

`make_results_table.py` renders `results_table.png` (or pass `--comparisons`
explicitly for a custom table). Rows group by `kp` so `casteer` and
`adaptive_kg` sit adjacent at matched gain — the only place the bold
best-in-block marker is a fair comparison, not best-in-column (which trivially
rewards whichever arm steers least).

### Single operating point (kp=2.0 only)

Only useful once the gain sweep above has established there's something to
compare at kp=2.0 — on its own this saturates Acc_e for every arm:

```bash
bash scripts/style/run_artist.sh vangogh 1 3 6 2
bash scripts/style/run_artist.sh kelly   3 6 1 4
bash scripts/style/score_artist.sh vangogh 3
bash scripts/style/score_artist.sh kelly   3
```

Both scorers run LPIPS in the generation environment and the VLM judge in
`vlmjudge` (Qwen2.5-VL needs `transformers >= 4.49`; the generation env is
pinned older). Override with `PYTHON=` / `VLM_PYTHON=`, and point `VLM_PATH` at
a local Qwen2.5-VL snapshot — it is loaded with `local_files_only=True`.

Single arm, by hand:

```bash
python scripts/style/generate_artist.py --artist "Van Gogh" \
    --concept_path "results/sd14/steering_vectors/Van Gogh.pt" \
    --controller adaptive_kg --kp 2.0 --ki 0.01 --kg 0.01 --kd 0.0 \
    --intermediate_clipping \
    --save_dir results/sd14/style/casteer_vectors/adaptive_kg_vangogh --gpus 1,3,6

python scripts/style/eval_artist.py lpips --artist "Van Gogh" \
    --baseline_dir results/sd14/style/baseline_vangogh/all \
    --method_dir   results/sd14/style/casteer_vectors/adaptive_kg_vangogh/all \
    --tag AdaptiveKg --append_to results/sd14/style/casteer_vectors/comparison_vangogh.json
```

`--diag` on a PID arm writes the per-block, per-step `P/I/G/D` trace (and
`adaptive_kg`'s `kg_weight`) to `{save_dir}/pid_records/{case_number}.csv`.

---

## Layout

```
scripts/style/
├── generate_artist.py       # all arms, protocol-pinned generation (called by everything below)
├── eval_artist.py           # lpips / acc-qwen / report
├── run_artist.sh            # single-point (kp=2.0) 4-arm run for one artist
├── score_artist.sh          # score run_artist.sh's output
├── run_gain_sweep.sh        # 24-arm kp/kd sweep; SV_SET=casteer|teca|heldout|retain picks the vector family
├── score_sweep.sh           # score run_gain_sweep.sh's output
├── run_overnight.sh         # chains the casteer + teca SV_SET sweeps + scoring + both tables
├── run_overnight_contrastive.sh  # builds Kelly's heldout/retain vectors, chains those two SV_SET sweeps + scoring + both tables
├── run_sweep_end_to_end.sh  # legacy, CASteer-vectors-only version of run_overnight.sh
├── make_results_table.py    # comparison_*.json -> results_table PNG + markdown
├── compare_steering_vectors.py  # cosine similarity between the two vector families
├── make_vector_grid.py      # side-by-side image grid, erased vs preserved bands
├── run_vector_ablation.sh   # 10-case grid comparing casteer vs TECA vectors at fixed kp
└── measure_vector_impact.py # LPIPS: does swapping the vector move the image more than the law?

scripts/diffusion/estimate_steering_vectors_style.py  # builds the heldout/retain .pt files (--negatives heldout|retain)
core/style_data.py           # prompt sets + the erased/unerased split
exp/datasets/eval/artists/   # big_artist_prompts.csv, short_niche_art_prompts.csv

results/sd14/style/
├── baseline_vangogh/ baseline_kelly/   # unsteered reference, shared by both vector sets
├── casteer_vectors/    # every arm built from results/sd14/steering_vectors/*.pt
│   ├── {casteer,pid,adaptive_kg}_{vangogh,kelly}/          # single-point (kp=2.0)
│   ├── casteer_kp{0.5,1.0,1.5,2.0}_{vangogh,kelly}/        # gain sweep
│   ├── adaptive_kg_kp{...}_kd{0.1,0.5}_{vangogh,kelly}/    # gain sweep
│   ├── comparison_{vangogh,kelly}.json          # single-point scores
│   ├── comparison_sweep_{vangogh,kelly}.json    # gain-sweep scores
│   └── results_table.png
├── teca_vectors/       # the same arms, built from SAFREE/style/sv_*_final_beta1.pt
│   ├── {casteer,pid,adaptive_kg}_teca_{vangogh,kelly}/     # 10-case vector-ablation grid
│   ├── casteer_teca_kp{...}_{vangogh,kelly}/               # gain sweep
│   ├── adaptive_kg_teca_kp{...}_kd{...}_{vangogh,kelly}/   # gain sweep
│   ├── comparison_sweep_teca_{vangogh,kelly}.json
│   └── results_table_teca.png
├── contrastive_vectors/  # heldout + retain arms, built from steering_vectors_{heldout,retain}/*.pt
│   ├── {casteer,adaptive_kg}_heldoutsv_kp{...}[_kd{...}]_{vangogh,kelly}/  # gain sweep
│   ├── {casteer,adaptive_kg}_retainsv_kp{...}[_kd{...}]_{vangogh,kelly}/   # gain sweep
│   ├── comparison_sweep_{heldout,retain}_{vangogh,kelly}.json
│   └── results_table_{heldout,retain}.png
├── vector_comparison/  # cosine analysis + band-split visual grid, cross-cutting both sets
│   ├── steering_vector_comparison.json
│   └── grid_vectors_{vangogh,kelly}.png
└── _archive/            # superseded outputs, not read by any current script
```

Every generated arm directory carries a `config.json` recording the
controller, the gains, the steering-vector file and the protocol constants
that produced it. Arm directory names keep a `_teca_` infix even inside
`teca_vectors/` (e.g. `casteer_teca_kp2.0_vangogh`) — redundant with the
parent folder, but it means a directory is still self-identifying if copied
out of context, and it's what `score_sweep.sh`'s glob/label logic keys on.

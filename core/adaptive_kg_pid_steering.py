"""
Ported into this repo from the PID experiment folder, unchanged except for
the import of `VectorControl`. Like `core.pid_steering_step_block`, it is a
drop-in `VectorControl`: the steering vectors, the hooked cross-attention
outputs and the set of steered blocks are exactly CASteer's -- only the
scalar applied to the steering direction differs. Select it with
`--controller adaptive_kg` in `scripts/diffusion/run_with_steering.py`.

The docstring below is the original one. Paths it names that are not in this
repo (`experiments/...`, `diag_pid_steering.py`, `run_unsafe_plus_safe.py`)
live in the PID folder this was ported from -- here the equivalent switch is
`--controller adaptive_kg` and the equivalent of `--diag` writes to
`{output_dir}/pid_records`. Its `core/pid_steering_step_block.py` references
do resolve, to the sibling module in this folder.

BLOCK-AWARE-Kg FORK, copied from this experiment's own diag_pid_steering.py
(itself a full copy of core/pid_steering_step_block.py -- see that file's
header). Not a subclass or import, so this never changes
pid_steering_step_block.py's own control math -- promoted into core/ as a
standing controller variant (selectable via run_unsafe_plus_safe.py's
--controller adaptive_kg), not just a one-off experimental prototype.

WHY THIS EXISTS
---------------
experiments/pid_global_ablation_nudity/results/block_adaptivity.png showed
Kg*I_global is EXACTLY FLAT across all 16 blocks: I_global(t) is a single
scalar shared by every block (by design, per core/pid_steering_step_block.py's
own docstring), so the global term pushes an already-clean block by the same
absolute amount as the most concept-loaded one. That block-blindness is a
plausible mechanism for the "ASR good, FID worse" symptom this experiment
was built to investigate: a uniform push adds correction to blocks that
don't need it, purely because OTHER blocks still carry the concept.

THE CHANGE
----------
Instead of a fixed, pre-chosen block set (the alternative the user
considered and rejected in favor of this), Kg's contribution is reweighted
PER BLOCK, PER STEP, by how that block's own current |error| compares to
the cross-block mean |error| from the last completed timestep (the same
"freeze at the timestep boundary" pattern _global_integral itself already
uses, extended to also cache the cross-block MEAN ABS error, not just the
signed mean):

    weight(b,t) = |e(b,t)| / mean_b'( |e(b',t-1)| )
    g_out(b,t)  = Kg * I_global(t) * weight(b,t)

A block far below last step's average error gets a correspondingly smaller
slice of the global push (weight < 1); one still above average gets more
(weight > 1). Averaged across blocks, weight ~= 1, so --kg_gain still sets
the OVERALL magnitude of the global term (comparable to the original at the
block-average level) -- this only changes how it's SPLIT across blocks, not
its total budget. `kg_weight` is logged to --diag for direct comparison
against the original's perfectly-flat Kg*I_global.

Also carries diag_pid_steering.py's mean_u_raw/mean_u_out fields (see that
file for why mean_abs_p/i/g/d_out alone can't answer "what is the actual
applied correction").

-------------------------------------------------------------------------
Original docstring (core/pid_steering_step_block.py) follows verbatim
-------------------------------------------------------------------------

Two-integrator PID steering: per-block history AND cross-block history.

This module combines the two integral axes that `core/pid_steering.py`
offers as mutually exclusive `--integral_axis {step,block}` choices. Both
run at once, with independent gains:

    e(b,t)   = <ca_out(b,t), b_hat(b)>                   # measured, setpoint 0
    u(b,t)   = Kp*e(b,t) + Ki*I(b,t) + Kg*I_global(t) + Kd*(e(b,t) - e(b,t-1))
    ca_out_new = ca_out - clamp(u(b,t), min=0) * b_hat

`core/pid_steering_step.py` is the Kg = 0 case of this law, and is
otherwise unchanged -- the P, I(b,t) and D terms here are computed
exactly as that module computes them.

-------------------------------------------------------------------------
The two integrators
-------------------------------------------------------------------------
I(b,t) -- BLOCK-SPECIFIC history, the timestep-axis integral:

    I(b,t) = sum_{j<t} e(b,j)

    One independent accumulator per (place_in_unet, block_index), running
    along the ~50 denoising steps. Shapes are constant at a fixed block,
    so this stays a full per-token tensor with no reduction. Answers "has
    THIS block had residual concept content for a while?"

I_global(t) -- GLOBAL history, the cross-block integral:

    I_global(t) = sum_{j<t} mean_b( e(b,j) )

    A single scalar shared by every block. At timestep j each steered
    block contributes its mean error, those are averaged across blocks,
    and that one number is folded into the running sum. Answers "does the
    network AS A WHOLE still carry the concept?" -- a block whose own
    error has gone quiet still gets pushed while other blocks have not.

Why the global term is frozen within a timestep: the control law indexes
it by t alone, not by (b,t), so every block visited during timestep t must
read the identical value. The timestep's per-block errors are therefore
buffered and only folded into I_global once the timestep closes (detected
on the first steered block visit of timestep t+1). The alternative --
accumulating on every block visit -- would make the value drift within a
timestep, so block 0 and block 15 of the same timestep would see different
I_global. That is deliberately not what this does.

Both integrators are summed over the same ~50 timesteps and both are sums
of the same error signal, so they reach comparable magnitudes and the same
`integral_clamp` anti-windup cap applies to each. Kg is therefore on the
same scale as Ki, not 16x smaller.

Only steered blocks enter the mean: unselected blocks return before any
state is touched, so a `selected_blocks` restriction narrows I_global to
the selected set rather than averaging in zeros. Changing the size of the
selection changes what I_global measures (it is a mean, not a sum, so its
magnitude does not scale with block count -- but its membership changes).

-------------------------------------------------------------------------
Gain scale
-------------------------------------------------------------------------
Ki ~ Kp/50 is the starting point, as in `core/pid_steering_step.py`, and
Kg starts at the same order as Ki. I_global is a mean over blocks, so it
sits near the magnitude of a typical block's own I(b,t) -- expect Kg*I to
land in the same range as Ki*I in the `--diag` readout. Run with diag=True
and read `pid_records/*.csv` before committing to a gain.

At Ki = Kg = Kd = 0 with clip=True this reduces to CASteer's Eq. 6 with
beta = Kp, bit-exactly: the error uses the same reshape/transpose/matmul
projection and the same cast-then-normalize ordering as CASteer's own
`steer_with_clipping`, and the zero-valued I/D/global terms are added as
exact floating-point zeros in the activation dtype.
"""
import torch

from .controller import VectorControl

VALID_PLACES = ('up', 'mid', 'down', 'joint', 'single', 'sana')


class AdaptiveKgPIDSteering(VectorControl):
    """PID steering with per-block and cross-block integrators, where the
    global term is reweighted per block/step by that block's own current
    error relative to last step's cross-block average (see module
    docstring). Also carries mean_u_raw/mean_u_out/kg_weight in --diag."""

    def __init__(
        self,
        *,
        source_concepts: list,
        target_concepts: list | None = None,
        kp_gain: float,
        ki_gain: float,
        kd_gain: float,
        kg_gain: float = 0.0,
        device,
        num_layers: int = None,
        clip: bool = True,
        integral_clamp: float | None = None,
        use_first_diffusion_step: bool = True,
        selected_blocks: set[tuple[str, int]] | None = None,
        diag: bool = False,
    ):
        super().__init__(num_layers=num_layers)

        if len(source_concepts) != 1:
            raise ValueError(
                f'PIDSteering tracks controller state per block and supports exactly one '
                f'concept; got {len(source_concepts)}. Use CASteer for multi-concept.'
            )
        if kp_gain < 0 or ki_gain < 0 or kd_gain < 0 or kg_gain < 0:
            raise ValueError('Negative gains are not supported')

        self.device = device
        self.kp_gain = kp_gain
        self.ki_gain = ki_gain
        self.kd_gain = kd_gain
        self.kg_gain = kg_gain
        self.clip = clip
        self.integral_clamp = integral_clamp
        self.use_first_diffusion_step = use_first_diffusion_step
        self.selected_blocks = selected_blocks
        self.diag = diag

        # Concept directions, built exactly as CASteer builds them
        # (target=None => erase toward zero). Deliberately NOT normalized
        # here: CASteer casts to the activation dtype and only then divides
        # by the norm, and reproducing that order is what keeps the
        # Ki=Kg=Kd=0 reduction bit-exact rather than 1-ULP off.
        source_concept = source_concepts[0]
        target_concept = (target_concepts or [None])[0]

        self.directions: dict[int, dict[str, list[torch.Tensor]]] = {}
        for num_steer in source_concept:
            self.directions[num_steer] = {}
            for place_in_unet in source_concept[num_steer]:
                block_dirs = []
                for block_idx in range(len(source_concept[num_steer][place_in_unet])):
                    source_vector = source_concept[num_steer][place_in_unet][block_idx]
                    if target_concept is not None:
                        target_vector = target_concept[num_steer][place_in_unet][block_idx]
                    else:
                        target_vector = torch.zeros_like(source_vector)
                    steering_vector = source_vector - target_vector

                    if steering_vector.dim() == 1:
                        steering_vector = steering_vector.unsqueeze(0)
                    block_dirs.append(steering_vector.to(self.device))
                self.directions[num_steer][place_in_unet] = block_dirs

        self._reset_pid_state()

    # ------------------------------------------------------------------
    # Controller state
    # ------------------------------------------------------------------
    def _reset_pid_state(self):
        self._accum: dict = {}          # I(b,t): sum of errors BEFORE the current one
        self._prev_error: dict = {}     # last error, for the derivative
        # I_global(t): scalar cross-block integral. Starts as a Python float
        # so the first `0.0 + tensor` fold adopts the activation dtype
        # (torch treats Python scalars as weak-typed), keeping the whole
        # control law in half precision.
        self._global_integral = 0.0
        self._step_errors: list[torch.Tensor] = []   # this timestep's per-block means
        self._step_abs_errors: list[torch.Tensor] = []  # this timestep's per-block |error| means
        # Cross-block mean |error| from the LAST COMPLETED timestep, used to
        # weight this timestep's Kg contribution per block (see module
        # docstring). 0.0 at t=0 is harmless: _global_integral is also still
        # 0.0 there, so g_out is 0 regardless of the weight's value.
        self._last_step_mean_abs_error = 0.0
        self._last_diffusion_step = None
        self._diag_stats = {'n': 0, 'p': 0.0, 'i': 0.0, 'g': 0.0, 'd': 0.0, 'e': 0.0}
        # Per-step P/I/G/D trace, kept separate from _diag_stats (which only
        # accumulates a single running mean). NOT cleared by reset(): the
        # caller drains it with drain_records() after each image, mirroring
        # DiagnosticProbe.drain_records() in diagnostic_steering.py.
        if not hasattr(self, 'records'):
            self.records: list[dict] = []

    def reset(self):
        super().reset()
        self._reset_pid_state()

    def _close_timestep(self):
        """Fold the finished timestep's across-block mean error into I_global,
        and cache its mean |error| for the NEXT timestep's Kg block-weights."""
        if not self._step_errors:
            return
        self._global_integral = self._global_integral + torch.stack(self._step_errors).mean()
        if self.integral_clamp is not None:
            self._global_integral = self._global_integral.clamp(-self.integral_clamp, self.integral_clamp)
        self._last_step_mean_abs_error = torch.stack(self._step_abs_errors).mean().item()
        self._step_errors = []
        self._step_abs_errors = []

    def _advance(self, key, error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (I, D) for this sample and fold `error` into the state.

        I(k) = sum_{j<k} e(j) -- excludes the current error, matching the
        paper's sum_{j=0}^{k-1}. D(k) = e(k) - e(k-1), zero at k=0.
        """
        prev = self._prev_error.get(key)
        accum = self._accum.get(key)

        # Shape guard: batch size can change between generations. Anything
        # that does not line up starts a fresh controller rather than
        # broadcasting silently.
        if prev is not None and prev.shape != error.shape:
            prev, accum = None, None
        if accum is not None and accum.shape != error.shape:
            accum = None

        i_term = accum if accum is not None else torch.zeros_like(error)
        d_term = (error - prev) if prev is not None else torch.zeros_like(error)

        new_accum = i_term + error
        if self.integral_clamp is not None:
            new_accum = new_accum.clamp(-self.integral_clamp, self.integral_clamp)
        self._accum[key] = new_accum
        self._prev_error[key] = error

        return i_term, d_term

    # ------------------------------------------------------------------
    # [batch_size, sequence_length, num_heads, head_dim]
    # ------------------------------------------------------------------
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        block = (place_in_unet, block_index)
        vector = vector.detach().clone()

        # Unselected blocks: a true no-op, nothing read and nothing tracked --
        # CASteer's own gating, unchanged. Placed before the timestep-boundary
        # check so I_global averages over steered blocks only.
        if self.selected_blocks is not None and block not in self.selected_blocks:
            return vector.half()
        if place_in_unet not in VALID_PLACES:
            return vector.half()

        # First steered block of a new timestep closes the previous one, so
        # every block within timestep t reads the same I_global(t).
        if diffusion_step != self._last_diffusion_step:
            self._close_timestep()
            self._last_diffusion_step = diffusion_step

        batch_size = vector.shape[0]
        # Steer only the prompt half of classifier-free guidance, as CASteer does.
        batch_slice = slice(batch_size // 2, None) if batch_size > 1 else slice(None, None)

        num_steer = 0 if self.use_first_diffusion_step else diffusion_step
        b = self.directions[num_steer][place_in_unet][block_index].to(dtype=vector.dtype)
        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)

        v = vector[batch_slice, ...]

        # e = <ca_out, b_hat>: measured concept content, setpoint 0.
        # Computed with the same reshape/transpose/matmul as CASteer's
        # steer_with_clipping -- an elementwise-multiply-and-sum gives the
        # same value but a different half-precision rounding, which breaks
        # the byte-exact Ki=Kg=Kd=0 equivalence check over 50 diffusion steps.
        bs, _, num_heads, hidden_dim = v.shape
        v_reshaped = v.to(self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        error = (v_reshaped @ b_norm.unsqueeze(-1)).transpose(0, 1).reshape(bs, -1, num_heads, 1)
        error = error.to(vector.device)
        b_norm = b_norm.to(vector.device)

        i_term, d_term = self._advance(block, error)
        # Contribute to I_global(t+1) and to next timestep's Kg block-weight
        # basis, NOT to the I_global/weight read below (both are frozen at
        # the values the LAST completed timestep produced).
        self._step_errors.append(error.mean())
        self._step_abs_errors.append(error.abs().mean())

        p_out = self.kp_gain * error
        i_out = self.ki_gain * i_term
        d_out = self.kd_gain * d_term
        # Block-aware global term (see module docstring): reweight the
        # shared I_global(t) by how this block's OWN current |error|
        # compares to the cross-block mean |error| from the last completed
        # timestep, instead of applying the identical scalar to every block.
        #
        # Before the first timestep boundary closes, _last_step_mean_abs_error
        # is still 0.0 and _global_integral is ALSO still 0.0, so g_out must
        # be 0 regardless of the weight -- computing the ratio anyway blows
        # up in fp16 (denominator ~1e-8 while error is a normal-sized value
        # routinely overflows fp16's ~65504 max to `inf`), and
        # `0.0 * inf = NaN` then poisons the activation for the rest of
        # generation. Skip the division entirely in that regime instead of
        # relying on the zero global_integral to cancel it out.
        if self._last_step_mean_abs_error > 0:
            # Capped at 10x a uniform share: still lets a loud block get
            # meaningfully more than a quiet one, without letting a
            # coincidentally tiny denominator (the concept nearly fully
            # erased for one timestep) spike the ratio unboundedly.
            kg_weight = (error.abs().mean() / (self._last_step_mean_abs_error + 1e-8)).clamp(max=10.0)
        else:
            kg_weight = torch.zeros((), dtype=error.dtype, device=error.device)
        g_out = self.kg_gain * self._global_integral * kg_weight
        u = p_out + i_out + d_out + g_out

        # Combined correction BEFORE clipping (diagnostic only -- see module
        # docstring). Gated on self.diag, matching the original's own
        # rationale for not syncing every visit when diagnostics are off.
        if self.diag:
            u_raw_mean = u.mean().item()

        # Never inject the concept -- the PID analogue of CASteer's
        # max(<ca_X, ca_out>, 0). At Ki=Kg=Kd=0 this is CASteer Eq. 6 exactly.
        if self.clip:
            u = torch.clamp(u, min=0)

        if self.diag:
            e_val = error.abs().mean().item()
            # Signed counterpart: |e| alone cannot tell over-correction (the
            # projection driven past zero) from under-correction, and the two
            # imply opposite fixes.
            e_signed = error.mean().item()
            # e is an UNNORMALIZED projection, so it rises either when the
            # concept grows or when the activation merely gets bigger. fp32
            # because the squared sum underflows/overflows more readily in half.
            ca_norm = v.float().norm(dim=-1, keepdim=True).to(error.device)
            norm_val = ca_norm.mean().item()
            cos_val = (error.float() / (ca_norm + 1e-6)).abs().mean().item()
            p_val = p_out.abs().mean().item()
            i_val = i_out.abs().mean().item()
            d_val = d_out.abs().mean().item()
            g_val = abs(float(g_out))
            s = self._diag_stats
            s['n'] += 1
            s['e'] += e_val
            s['p'] += p_val
            s['i'] += i_val
            s['g'] += g_val
            s['d'] += d_val
            self.records.append({
                'diffusion_step': diffusion_step,
                'place_in_unet': place_in_unet,
                'block_index': block_index,
                'mean_abs_e': e_val,
                'mean_e_signed': e_signed,
                'mean_ca_norm': norm_val,
                'mean_abs_cos': cos_val,
                'mean_abs_p_out': p_val,
                'mean_abs_i_out': i_val,
                'mean_abs_g_out': g_val,
                'mean_abs_d_out': d_val,
                'i_global': float(self._global_integral),
                'mean_u_raw': u_raw_mean,
                'mean_u_out': u.mean().item(),
                # NaN before the first timestep boundary closes (denominator
                # is still 0): the ratio is numerically huge there but never
                # matters for the control law (i_global is also still 0), so
                # don't let it pollute a kg_weight plot with a meaningless spike.
                'kg_weight': float(kg_weight) if self._last_step_mean_abs_error > 0 else float('nan'),
            })

        vector[batch_slice, ...] = v + (-u) * b_norm
        return vector.half()

    def drain_records(self) -> list[dict]:
        """Pop and return the per-step P/I/G/D trace accumulated since the last drain.

        Only populated when diag=True. Call after each generated image (and
        before the next reset()) to get one image's trajectory without it
        bleeding into the next.
        """
        records, self.records = self.records, []
        return records

    def diag_summary(self) -> str:
        s = self._diag_stats
        if not s['n']:
            return 'no steered blocks visited'
        n = s['n']
        return (
            f"mean|e|={s['e']/n:.4f}  "
            f"mean|Kp*e|={s['p']/n:.4f}  "
            f"mean|Ki*I|={s['i']/n:.4f}  "
            f"mean|Kg*Ig|={s['g']/n:.4f}  "
            f"mean|Kd*D|={s['d']/n:.4f}  "
            f"(over {n} steered block visits)"
        )

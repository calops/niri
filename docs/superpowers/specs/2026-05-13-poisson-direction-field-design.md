# Poisson-equation direction field for subregion masks

**Status:** Active design. Replaces the SDF-mipmap direction path; the
JFA-derived scalar SDF stays for the R channel of the output. See
`2026-05-13-vector-field-blur-future-work.md` for the cheaper alternative we
chose to skip.

## Why this exists

Every direction-field strategy we tried so far computes a *scalar* field
(SDF, blurred SDF, density) and takes its gradient. They all share the same
fundamental defect: even after heavy smoothing, the scalar field still has
maxima at the medial axis, and the gradient near a maximum *has to* rotate
fast. Combined with the unit-length renormalisation the renderer needs, this
shows up as visible seams along the medial axis.

The mathematically correct fix is to solve a PDE that produces a scalar field
with **smooth gradient everywhere inside the shape**. The classical choice is
the screened Poisson / heat equation with a Dirichlet boundary at the shape
edge.

## The math

Solve, over the interior of the binary mask `Ω`:

```
∇²u = -1   inside Ω
u    = 0   on ∂Ω
```

The solution `u(x, y)` is the **torsion function** / **Saint-Venant
function** of the domain. It's strictly positive in the interior, zero on the
boundary, has a single maximum somewhere near the "thermal centre" of `Ω`,
and crucially: **`∇u` is smooth everywhere in `Ω`**, with no medial-axis
ridges (proven from elliptic regularity theory).

For a circular disk of radius `R` this gives exactly the analytical
`(R² − r²) / 4`, a perfect dome with linear-radial gradient — i.e. the
direction field becomes `-r̂`, the analytical "to-centre" field we want for
the full-window case. For a rectangle it's a smooth quasi-radial field, no
diagonal seams. For arbitrary shapes (including 1-px-column-built
triangles, rounded corners, etc.) the gradient is smooth by construction.

The encoded output:
- `R` channel: SDF from JFA, as today (depth from boundary)
- `GB` channel: `dir = normalize(∇u)` from the Poisson solution

## Why it's tractable on GPU

Naïve Jacobi iteration converges in `O(N)` iterations on an `N×N` domain —
too slow. The right algorithm is **geometric multigrid** with a V-cycle:

- Restrict the residual through a pyramid (~5 levels)
- Smooth (a few Jacobi sweeps) at each level
- Prolongate the correction back up the pyramid
- Repeat the whole V-cycle 1–2× per frame

Convergence is `O(log N)` cycles, and most of the work happens at low
resolution. For a 300×300 bbox this is roughly 30–50 fragment-shader passes,
well under 1 ms wall-time on typical GPUs. For a 1500×1500 region it's still
under 1 ms because the pyramid does the work at lower resolutions.

Multigrid is exactly the same pyramid infrastructure we already render for
the mipmap-SDF approach; only the schedule of operations on it is different.

## Implementation sketch

**Shaders:**

- `jfa_poisson_jacobi.frag` — one Jacobi sweep at a given pyramid level:
  ```
  u_new[i,j] = mask[i,j] ? 0.25 * (u_left + u_right + u_up + u_down + h² * f) : 0
  ```
  where `f = 1`, `h` is the texel spacing at the current level, and `mask`
  is the binary mask resampled to that level. Boundary handling is the
  multiplication by `mask` (Dirichlet `u = 0` outside the shape).

- `jfa_poisson_restrict.frag` — compute residual `r = f − Lu` and downsample
  it by box-filter for the next-coarser level. Outputs `r` to the
  next-level RHS texture. Also resamples the binary mask to the coarser
  level via majority-vote or alpha-weighted average.

- `jfa_poisson_prolongate.frag` — upsample the coarse-level correction with
  bilinear filter and add it to the fine-level solution.

- `jfa_poisson_gradient.frag` — central-difference gradient of the final
  Poisson solution, normalised and packed into G/B. (Equivalent to the
  current encode-direction step, just sampling the Poisson field instead
  of the mipmapped SDF.)

**Textures (per pyramid level k = 0..K):**

- `u[k]` — current solution at level k
- `f[k]` — right-hand side at level k (`f[0] = 1` inside mask, refined by
  `restrict` for k > 0)
- `mask[k]` — binary mask at level k

**Orchestration (Rust):**

1. Initialise `u[0] = 0` (or seed from previous frame for temporal
   coherence).
2. For each V-cycle:
   - Down sweep: for k = 0..K-1, smooth `u[k]` a few times, compute
     residual, restrict to `f[k+1]`.
   - Coarsest level: solve exactly (1×1, trivial).
   - Up sweep: for k = K..1, prolongate correction from `u[k]` into
     `u[k-1]`, smooth a few times.
3. After 1–2 V-cycles, take gradient of `u[0]` → direction field.

Typical schedule: 5 levels, 3 pre-/post-smoothing sweeps each, 1–2 V-cycles
per frame.

## What stays, what goes

Stays:
- `mask_binary.frag` — produces the binary mask (also reused as `mask[0]`
  for the multigrid solver).
- `jfa_init.frag`, `jfa_step.frag` — JFA stages, still used for the R
  channel.
- `jfa_sdf_bake.frag` — bakes the JFA result into a scalar SDF in R; output
  becomes the R channel of the final `encoded` texture.

Goes:
- `jfa_sdf_downsample.frag` and the SDF mipmap chain — replaced by the
  multigrid pipeline.
- `JfaSdfDownsampleProgram` and the textureLod-based gradient logic in
  `jfa_encode.frag`.

Net: the encode pass now reads two textures: the JFA-baked scalar SDF (for
R) and the Poisson solution `u[0]` (for ∇u → GB).

## Initial values and defaults

- Pyramid depth `K = 5` levels (bbox/32 at the coarsest).
- 3 Jacobi smoothing sweeps before restriction, 3 after prolongation per
  level.
- 1 V-cycle per frame initially. Promote to 2 if visual quality requires
  it.
- `u[0]` reset to 0 each frame initially. Promote to temporal warm-start
  if 1 V-cycle isn't sufficient.
- Weighted (damped) Jacobi with `ω = 4/5` — standard choice for multigrid
  smoothers, gives better high-frequency damping than plain Jacobi.

## References

- Saad, Y. *Iterative Methods for Sparse Linear Systems*, Chapter 13
  (Multigrid).
- Crane, K., Weischedel, C., Wardetzky, M. (2017). *The heat method for
  distance computation*. (Same kind of PDE infrastructure, different
  boundary conditions and right-hand side.)
- McAdams, A., Sifakis, E., Teran, J. (2010). *A parallel multigrid Poisson
  solver for fluids simulation on large grids*. SCA. — GPU implementation
  patterns, including red-black smoothing schedules.

## Open questions

- Temporal coherence: can we warm-start `u[0]` from the previous frame's
  solution to converge in fewer cycles when the shape is animating?
- Boundary anti-aliasing: should `mask[k]` be a binary mask resampled with
  majority vote, or a soft alpha mask? The latter gives anti-aliased
  boundaries in the gradient but complicates the Jacobi update.
- Per-frame V-cycle count: fixed (1–2) or adaptive based on residual norm?

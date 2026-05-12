# Future work: Vector-field blur for subregion direction

**Status:** Deferred. Skipped in favour of Poisson multigrid (see
`2026-05-13-poisson-direction-field-future-work.md`, since promoted to the
active design). Revisit if Poisson multigrid turns out to be overkill and we
want a cheaper alternative.

## The idea

Instead of computing a *scalar* SDF, smoothing it, and taking its gradient
(which produces medial-axis seams in the gradient direction), produce a
**vector field** directly and smooth *that*.

For each interior pixel, the natural inward attractor is
`v(p) = normalize(p − nearest_exterior_pixel)` — a unit vector pointing
away from the closest boundary. This is the negative SDF gradient. We
already have `nearest_exterior_pixel` from JFA.

Naively this field has discontinuities at the medial axis (different sides
of the axis pick different "nearest" pixels). But if we *blur the unit
vectors* rather than the scalar SDF, the discontinuities cancel into a
smooth field:

- At the centre of a square the kernel sees `up + down + left + right ≈ 0`
  → small magnitude, smooth.
- One step toward an edge the kernel skews toward the perpendicular of
  that edge → soft inward vector.
- Near a corner, kernel sees a blend of two perpendiculars → smooth
  rotation.

## How it would map onto our infrastructure

Almost identical to the current mipmap-SDF code path, just with a
two-channel payload:

1. **Bake** — write `(p − nearest) / max_dist` into RG channels (instead of
   `length(...)` into R alone).
2. **Mipmap chain** — same `jfa_sdf_downsample.frag`, but reading RG
   instead of R. Box-filter average of vectors.
3. **Encode** — sample the mipmapped RG at per-pixel LoD, take the result
   as the direction (no central-difference gradient needed; the field
   *is* the direction). R channel still comes from JFA scalar distance.

Net change vs. current code: ~one shader rewrite + a couple of uniform/
sampler swaps. Maybe 30 minutes of work.

## Why we deferred it

- Mathematically heuristic. The average of unit vectors is not equivalent
  to any specific PDE solution; it's an artist's smoothing. For convex-ish
  shapes it should look great, but on intricate concave shapes
  (e.g. user's 1-px-column-built triangles or rounded corners) it might
  still produce visible artefacts where the kernel straddles complex
  boundary geometry.
- The Poisson multigrid approach gives the mathematically correct smooth
  field for the same infrastructure cost (~3–4 hours instead of 30 min),
  and is robust to arbitrary shapes. We chose to bypass the cheap
  experiment.

## When to revisit

- If Poisson multigrid implementation is much more painful than expected.
- If the per-frame cost of Poisson multigrid is too high in practice.
- If we find a use case where the artist-style heuristic actually looks
  *better* than the mathematically correct field (this happens sometimes
  in stylised rendering).

## Open question for revisit

Should magnitude be preserved or always renormalised? Vector blur naturally
shortens vectors near the medial axis (where opposing directions cancel),
which the renderer can read as "weaker glass curvature here". This may be
desirable or undesirable depending on artistic intent.

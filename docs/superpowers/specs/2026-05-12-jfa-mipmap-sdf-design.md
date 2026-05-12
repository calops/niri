# Subregion mask via mipmapped SDF

## Goal

Produce the SDF + direction texture consumed by custom-blur pipelines (e.g.
`overshifted3`) for arbitrary subregion shapes, with a *smooth* interior vector
field that has no medial-axis ridges and no seams, while still preserving sharp
direction information near complex boundary details. The result for a plain
rectangle should look indistinguishable from the analytical full-window SDF
path.

The full-window analytical SDF path in `mask.frag` is **unchanged**.

## Why the previous design failed

The earlier plan computed direction from the gradient of a separable box-blurred
field. Two failure modes were observed:

- Blurring the **binary mask** gives a gradient that exists only in a thin
  band near the boundary (blur radius wide). The interior is flat `1.0` and has
  zero gradient → no direction info.
- Blurring the **distance field** gives a gradient everywhere, but the
  blur (5–20 px) is far too small to span the medial axis. The result is
  Voronoi-style ridges meeting at the centroid — visible as sharp seams along
  the diagonals of a rectangle.

Cranking the blur radius to be comparable to the shape's half-extent (50–100 px)
would smooth the ridges out, but a uniform large blur erases small boundary
detail in complex shapes.

## Strategy: variable-radius blur via mipmaps

For each interior pixel, pick a blur radius equal to its **distance to the
boundary**: small near the edge (preserves boundary detail), large deep inside
(smooths the medial axis where the smear is invisible anyway).

The GPU-efficient form is mipmap + per-pixel LoD sampling: at each mip level
*k*, the texture is downsampled by 2ᵏ, which is mathematically equivalent to a
box blur of radius ~2ᵏ pixels. Sampling at fractional LoD with linear filtering
interpolates smoothly between adjacent levels.

Result per pixel: the gradient of the SDF *seen at the appropriate scale for
that pixel*. Near edges this is the sharp local SDF gradient. Deep inside it's
the gradient of a heavily-smoothed field that has no ridges.

This is the same idea that drives SSAO and depth-of-field. Cost is one mipmap
chain (essentially free) plus 4 textureLod samples per pixel in the encode
shader.

## Output contract (unchanged)

Same as the previous designs, so renderers don't change:

- `R` — normalized SDF, `0` at boundary → `1` at deep interior. Exterior pixels
  write `(0, 0.5, 0.5, 1)` so the existing `mask < 0.001` early-out triggers.
- `G`, `B` — `dir * 0.5 + 0.5`, where `dir` is a unit-ish vector pointing
  inward. Renderers decode as `(gb - 0.5) * 2.0`.

## Pipeline stages

All stages run in bbox space.

1. **Binary mask** (`mask_binary.frag`, existing, unchanged).
2. **JFA init** (`jfa_init.frag`, existing, unchanged).
3. **JFA steps** (`jfa_step.frag`, existing, unchanged). Power-of-two ping-pong
   on `textures.jfa_a` / `textures.jfa_b`.
4. **SDF bake** (NEW, `jfa_sdf_bake.frag`). One pass that reads the JFA
   result and writes a scalar SDF texture: `frag_color.r = clamp(dist /
   max_dist, 0, 1)` where `dist = length(pixel - nearest_exterior_coord)`.
   Exterior pixels write `0`. Output: `textures.sdf` (RGBA16F, only R used).
5. **Mipmap generation** (NEW, no shader — single GL call). `glGenerateMipmap`
   on `textures.sdf`. Texture is set up to support mip storage; min-filter is
   `LINEAR_MIPMAP_LINEAR`, mag-filter `LINEAR`.
6. **Encode** (rewritten, `jfa_encode.frag`). Reads the JFA result (for the R
   channel) and the mipmapped SDF (for direction):
   - SDF distance `dc = length(pixel - nearest)` from JFA. Exterior →
     sentinel.
   - `mask = clamp(dc / niri_max_dist, 0, 1)` for R channel.
   - LoD selection: `lod = clamp(log2(max(dc, 1.0) / niri_lod_base), 0,
     niri_max_lod)`. Logarithmic with depth — a pixel `2ᵏ * niri_lod_base` from
     the boundary samples at mip k. `niri_lod_base` is a tunable uniform
     (default ~4 px), controlling how aggressively blur scales with depth.
   - Gradient via central differences with `textureLod(niri_sdf_mip, uv ±
     step, lod)`. Step size is `1 / niri_output_size` (sample at adjacent
     mip-0 texels; linear filtering at the chosen LoD does the rest).
   - Normalize, encode into G/B.

The `jfa_density_blur.frag` shader and the density / density_h textures are
**removed**.

## Texture layout

| Slot | Purpose | Format | Mipmaps |
|---|---|---|---|
| `bin` | Binary mask | RGBA16F (`Fourcc::Abgr16161616f`) | no |
| `jfa_a`, `jfa_b` | JFA ping-pong | RGBA16F | no |
| `sdf` | Scalar SDF, mipmapped (R channel only) | RGBA16F | yes |
| `encoded` | Final SDF + direction | RGBA16F | no |

Five textures total (down from six in the previous plan; we drop `density_h`
and `density`, add `sdf`).

The SDF only uses one channel of `sdf` but we stick with RGBA16F for two
reasons: (1) Smithay's `create_buffer` only maps `Fourcc::Abgr8888` and
`Fourcc::Abgr16161616f` to GL formats — single-channel R8/R16 would require
bypassing the smithay API and creating the texture by hand. (2) 8-bit
normalized values band visibly at deep mip levels of an SDF, where per-texel
differences shrink below quantization. RGBA16F costs ~1 MB for a typical
500×500 bbox texture; acceptable.

For mipmap storage: Smithay's `create_buffer` allocates `glTexImage2D`-style
mutable storage with only level 0. `glGenerateMipmap` will lazily allocate the
rest of the chain on first call, provided the min-filter is set to a mipmap
variant beforehand. If that proves unreliable, the fallback is manual texture
creation via `glTexStorage2D(levels, ...)` — but this requires bypassing
`create_buffer`, so we try the simpler path first.

## LoD scaling

The `niri_lod_base` uniform controls how quickly the blur grows with depth.
With `niri_lod_base = 4`:
- 0–4 px from boundary → mip 0 (no blur, sharp local gradient)
- 4–8 px → mip 1 (≈2 px blur)
- 8–16 px → mip 2 (≈4 px blur)
- 16–32 px → mip 3 (≈8 px blur)
- 32–64 px → mip 4 (≈16 px blur)
- 64+ px → mip 5+ (≈32+ px blur)

`niri_max_lod` is clamped to the available mip count, typically
`floor(log2(min(bbw, bbh)))`.

Both are uniforms so they can be tuned without rebuilding shaders.

## Rust changes (`src/render_helpers/blur.rs`)

- Replace `JfaDensityBlurProgram` with `JfaSdfBakeProgram` (uniforms:
  `niri_input` for JFA, `niri_output_size`, `niri_max_dist`).
- Add new uniforms to `JfaEncodeProgram`: `niri_sdf_mip` (sampler2D),
  `niri_lod_base`, `niri_max_lod`. Drop `niri_density`.
- Replace `JfaTextures` fields `density_h` / `density` with `sdf`.
- Replace `compute_density` stage with `bake_sdf` + `generate_sdf_mipmaps`.
- `encode_output` binds `textures.sdf` on TEXTURE1, sets the new uniforms,
  and uses `textureLod`-based gradient sampling (this is a shader change; the
  Rust call site changes only via uniform setup).

The named-stage refactor from the previous plan is preserved.

## Non-goals

- The analytical full-window SDF path (`mask.frag`) is untouched.
- The renderer pipelines (`overshifted3/*.frag`, `jfa-debug/viz.frag`) are
  unchanged.
- The Kawase blur in `Blur::render` (non-custom path) is unchanged.

## Validation

- `cargo check` builds clean.
- Visual: with `overshifted3` on a single full-window region the result
  should look like the analytical SDF case (smooth radial, no seams).
- Visual: with `jfa-debug`, the R channel should ramp smoothly from 0 at
  edges to ~1 at the deepest interior point; the GB direction should be
  smooth across the entire interior with no visible ridges along diagonals
  of a rectangle.
- Visual: complex multi-rect subregion configurations should show smooth
  direction in the interior while preserving sharp gradient at the
  rectangle joints and any thin features.

## Decisions

- `niri_lod_base = 4` is set as a shader uniform in `blur.rs`. Not exposed in
  config; can be promoted later if needed.
- `Fourcc::Abgr16161616f` for the SDF texture (R channel only). Single-channel
  formats aren't supported by smithay's `create_buffer` and 8-bit risks
  banding at deep mip levels.

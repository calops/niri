# JFA mask pipeline redesign

## Goal

Produce the SDF+direction texture consumed by custom-blur pipelines (e.g. `overshifted3`) for arbitrary subregion shapes, with a credible glass look: smooth interior curves without losing boundary sharpness.

The full-window analytical SDF path (in `mask.frag`) is unchanged. Only the JFA-based subregion path is redesigned.

## Output contract

A single RGBA16F texture, sampled by `glass.frag` / `refract.frag` / debug viz as `niri_mask`:

- `R` — normalized signed distance, `0` at boundary → `1` at deepest interior. Exterior samples write `(0, 0.5, 0.5, 1)` so the existing `mask < 0.001` early-out continues to work.
- `G`, `B` — encoded 2D vector `dir * 0.5 + 0.5`, where `dir` is a unit-ish vector pointing inward (toward interior). Renderers decode as `(gb - 0.5) * 2.0`.

This matches the analytical SDF output, so renderer code is untouched.

## Pipeline stages

All stages run in **bbox space** (a padded bounding rectangle around the union of subregion rects). The final encoded texture is then blitted into the source-resolution mask texture at the bbox location.

1. **Binary mask** — `mask_binary.frag`. R=1 inside any subregion rect, R=0 outside. (Existing, unchanged.)
2. **JFA init** — `jfa_init.frag`. Exterior pixels seed with their own pixel coords; interior pixels write `(-1, -1)`. (Existing, unchanged.)
3. **JFA steps** — `jfa_step.frag`. Power-of-two ping-pong (start at `step = largest_pow2 ≤ max_dim`, halve until 0). Each interior pixel converges to the nearest exterior seed coord. (Existing, unchanged.)
4. **Density blur** (NEW) — `jfa_density_blur.frag`. Separable box blur over the binary mask, run twice (horizontal pass → `density_h`, vertical pass → `density`). Packs two radii in one texture per pass: `R = density_low` (radius `niri_radius_low`, default 2px), `G = density_high` (radius `niri_radius_high`, default 20px). Both radii are uniforms for tweaking. Inner loop is unrolled over a `const int MAX_R` with a `niri_radius` cutoff (one cutoff per channel).
5. **Encode** (REWRITTEN) — `jfa_encode.frag`. Single pass reading the JFA result and the density texture:
   - Compute SDF distance `dc` from JFA nearest-coord; if `dc < 0` (exterior sentinel) emit the exterior fallback.
   - Normalize: `mask = clamp(dc / niri_max_dist, 0, 1)` where `niri_max_dist = min(bbw, bbh) / 2` so deep interior reads near 1 regardless of bbox aspect.
   - Compute central-difference gradients of `density_low` and `density_high`.
   - Blend: `t = smoothstep(0, niri_edge_threshold_px, dc)`, `dir = mix(grad_low, grad_high, t)`, then normalize.
   - Emit `vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1)`.
   - `niri_edge_threshold_px` defaults to `20.0` (matches the high-density radius). Uniform for tweaking.

The dead `jfa_blur.frag` is deleted.

## Texture layout

All textures are bbox-sized, RGBA16F. Allocated once per bbox-size change, named, no reused indices.

| Slot | Purpose |
|---|---|
| `bin` | Binary mask (output of stage 1, input to JFA init + density blur) |
| `jfa_a`, `jfa_b` | JFA ping-pong |
| `density_h` | Density horizontal intermediate (R=low_h, G=high_h) |
| `density` | Density final (R=low, G=high) |
| `encoded` | Final SDF+direction texture; blitted into the source-resolution mask texture |

Six textures total (currently 4). Memory delta is small — all are bbox-sized.

## Rust-side changes (`src/render_helpers/blur.rs`)

- Replace the monolithic `render_jfa_mask` block with named stage functions called in sequence: `render_binary_mask`, `run_jfa`, `compute_density`, `encode_output`, `blit_to_mask_texture`. Each takes the GL context and explicit input/output texture references — no shared `read_idx` / `write_idx` index dance.
- `JfaPipeline` gains a `density_prog: JfaDensityBlurProgram` field. `JfaEncodeProgram` gains uniform locations for `niri_density`, `niri_edge_threshold_px`, `niri_max_dist` (latter exists). Density program has `niri_input`, `niri_output_size`, `niri_axis`, `niri_radius_low`, `niri_radius_high`.
- Texture array `self.jfa_textures` grows from 4 to 6. Allocate by name into a small struct rather than by index.

## Non-goals

- Renderer pipelines (`overshifted3/*`, `jfa-debug/viz.frag`) are not touched except optionally to improve viz of the new fields.
- Analytical-SDF full-window path is unchanged.
- The downsample/blur portion of `Blur::render` (non-custom) is unchanged.

## Validation

- Build cleanly (`cargo build`).
- Run with `overshifted3` configured on a multi-region window: visually check smooth interior curves and boundary sharpness.
- `jfa-debug` pipeline can be used to visualize R / GB channels.

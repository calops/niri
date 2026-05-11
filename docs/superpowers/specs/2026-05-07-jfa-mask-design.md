# Jump Flooding Mask Pipeline — Design

## Overview

Replace the per-rectangle analytical SDF mask (limited to 16 uniform-bound subregions) with a
Jump Flooding Algorithm (JFA) distance field computed from an arbitrary binary mask. The
full-window case (no subregions) retains the existing analytical SDF for zero-overhead rounded
rectangles. This removes the 16-region limit, eliminates rectangle merging, and handles
arbitrary shapes natively.

## Decision Tree

```
subregion_count == 0  →  analytical SDF (current mask.frag, unchanged)
subregion_count  > 0  →  binary mask → multi-scale JFA → distance field → encode
```

Both paths write to the same `niri_mask` RGBA16F texture with identical channel layout:
- **R:** mask intensity (0 = outside, 1 = deepest interior)
- **G:** `to_nearest.x * 0.5 + 0.5` (direction to nearest boundary point)
- **B:** `to_nearest.y * 0.5 + 0.5`
- **A:** `1.0` (reserved)

Consumer shaders (refract, magnify, glass) are unchanged.

## JFA Pipeline — Per Frame

All passes output to intermediate RGBA16F textures. The final output replaces the current
mask texture (`self.mask_texture` in `Blur`).

### Pass 1: Binary Mask

Input: subregion rectangles (arbitrary count, no 16-limit).
Output: RGBA16F texture at the union bounding box resolution.

Shader: iterate over rects (passed as a dynamic uniform array, GLES 3.1+, or SSBO),
fill each as white (`1.0`), leave rest black (`0.0`).

> **Provision: bounding-box JFA.** All JFA passes run at the union bounding box
> resolution, NOT the full window size. For a 1920×1080 window with two 60×400 bars
> at positions (0,200) and (1860,200), the binary mask is 1860×600 (the bounding
> box of both bars), not 1920×1080. The final encode pass pads the result into the
> source-sized mask_texture, writing zero outside the union box.

### Pass 2: Downscale

Bilinear downscale to 1/4 of the binary mask resolution. This becomes the seed
resolution for JFA.

### Pass 3: JFA Seed

Initialize the distance field at the downscaled resolution.
For each pixel:
- If the binary mask sample is > 0.5: `seed = (pixel_x, pixel_y)` in pixel coordinates
- Else: `seed = (∞, ∞)` (sentinel, encoded as negative or max-float values)

Output: texture where each pixel stores `(seed_x, seed_y)` (the nearest known interior point).

### Passes 4-N: JFA Steps

Logarithmically decreasing step sizes. For a seed texture of size W×H, step sizes are
`max(W,H)/2, max(W,H)/4, ..., 1`. Each pass reads the previous pass's seed texture and
propagates seeds using a fixed 8-neighbor jump pattern.

Shader (uniform per pass): `step_size` (pixel offset), `seed_texture` (previous pass).

For each pixel at `(x, y)`:
1. Read the current best seed: `best = texture(prev, uv).rg`
2. For each neighbor offset `(dx, dy)` in `{+step, 0, -step}²`:
   - Read neighbor seed: `cand = texture(prev, uv + offset/tex_size).rg`
   - If `length(cand - pixel) < length(best - pixel)`: `best = cand`
3. Write `best` to output

Output: updated seed texture (closer to true nearest interior point).

### Pass N+1: Upscale

Bilinear upscale the seed texture back to the full bounding box resolution.

### Passes N+2, N+3: Refinement

Two additional JFA passes at full resolution with step sizes 2 and 1, seeded from the
upscaled result. This recovers boundary details lost during downscale.

### Final Pass: Encode

Convert the seed texture to the canonical `niri_mask` format:
```glsl
vec2 nearest = texture(seed_tex, uv).rg;
float dist = length(nearest - pixel_coord);
float mask = dist / max_dist; // max_dist precomputed or derived
vec2 to_nearest = nearest - pixel_coord;
```

Write to `self.mask_texture` in the standard RGBA16F format.

### Pass Count Summary

| Step | Passes | Resolution |
|------|--------|------------|
| Binary mask | 1 | union box |
| Downscale | 1 | 1/4 union box |
| JFA seed | 1 | 1/4 union box |
| JFA steps | ~log₂(max_dim/4) | 1/4 union box |
| Upscale | 1 | union box |
| Refinement | 2 | union box |
| Encode | 1 | union box |

Example: 1920×1080 union box → 480×270 downscale → 6 JFA steps → ~13 total passes.

## Rust Changes

### BlurOptions

Add `subregion_count` field (replaces reliance on `subregion_rects.len()`).

Remove `subregion_rects` (no longer passed as shader uniforms — used only by binary mask pass).

### Blur::render_custom

Branch on `options.subregion_count`:
- `== 0`: run existing analytical mask shader (unchanged code path)
- `> 0`: run JFA pipeline (new code path)

The JFA path allocates intermediate textures and dispatches the passes described above.
The final output goes to `self.mask_texture` in the same format, so the downstream
`render_custom` loop is unchanged.

### FramebufferEffect

Remove the subregion rect merging logic (the `MAX_BLUR_SUBREGIONS`-capped merging
into `subregion_rects`). Rects now pass through directly for the binary mask render.

### Mask Shaders

New shader files under `src/render_helpers/shaders/`:
- `mask_binary.frag` — render rects to binary mask
- `jfa_init.frag` — seed initialization at downscaled resolution
- `jfa_step.frag` — single JFA propagation step
- `jfa_encode.frag` — final encoding to canonical mask format

Downscale and upscale reuse existing hardware bilinear filtering (no custom shader needed).

The existing `mask.frag` remains for the `subregion_count == 0` analytical SDF path.

### Constants Removed

- `MAX_BLUR_SUBREGIONS` (16) — no longer needed
- `niri_subregion_count` uniform from mask.frag — only used in the zero-subregion path,
  can be replaced with a simple boolean or the existing flow
- `niri_subregion_rects` uniform from mask.frag — no longer needed

## Consumer Shaders

No changes. The `niri_mask` texture format is identical:
```glsl
vec4 m = texture(niri_mask, uv);
float mask = m.r;
vec2 to_center = (m.gb - 0.5) * 2.0;  // now points to nearest boundary, not rect center
```

The semantic change: `to_center` now points toward the nearest boundary point (the JFA seed),
which for convex regions points approximately toward center. For concave or complex shapes,
the field follows the true Euclidean distance gradient — which is the physically correct
behavior for glass refraction.

## Constraints & Edge Cases

- **GLES version:** The binary mask shader needs dynamic iteration over rects. Requires
  GLES 3.1+ (for SSBO) or we pass rect data as a 1D texture.
- **Empty subregions:** Handled naturally — JFA seeds mark all pixels as empty, final
  mask is all-zero.
- **Single tiny subregion:** JFA at 1/4 scale still works. ~6 JFA passes even for tiny boxes.
  The overhead vs. analytical SDF is the motivation for keeping the SDF path.
- **Mask texture size:** The JFA output replaces the existing `mask_texture` at the union
  box size, not at the source texture size. The mask is sampled by effect passes, which
  use the same UV coordinates — the mask texture must be the same size as the source.
  We either create the JFA output at source-size, or rescale during encode.

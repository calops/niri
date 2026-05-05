# GPU Mask Pipeline Design

## Problem

Current mask generation runs EDT on CPU at half resolution. Issues:
- Half-resolution artifacts visible near edges where effects are strongest
- CPU EDT cost (~8ms full res, ~2ms half res) hurts on resize at high refresh rates
- Bbox optimization helps subregions but not full-window resize

## Approach

Move mask generation to GPU as fragment shader passes at native resolution.
Two paths, selected by overlap detection (O(N²) on N≤16 rects, instant):

1. **Cheap path (no overlaps)**: 1 pass — per-pixel axis-aligned distance `min(|x-edges|, |y-edges|)` + to_center direction. Trivial.
2. **JFA path (overlaps)**: ~14 passes — seed binary mask, 12 JFA passes, final normalize + direction.

Cache: skip entire mask pipeline when rects + size unchanged (same as now).

## Data flow

```
Rust: detect overlaps, check cache

No overlaps:
  GPU: mask_cheap → down → up → refract → glass

Overlaps:
  GPU: mask_seed → JFA[0..11] → mask_final → down → up → refract → glass
```

## Shaders

### mask_cheap.frag (1 pass)
- Input: subregion rects via uniforms
- Output: RGBA8 at full res (R=distance, GB=to_center, A=255)
- Logic: for each rect, `d = abs(uv - edges)`, take min; direction = toward nearest rect center
- Full-window (0 rects): `d = min(uv.x, 1-uv.x, uv.y, 1-uv.y)`, dir = toward (0.5, 0.5)

### mask_seed.frag (1 pass)
- Input: subregion rects via uniforms
- Output: R32F or RGBA8 binary mask at full res (1.0 inside, 0.0 outside)
- Same fill logic as current CPU binary fill

### mask_jfa.frag (~12 passes, ping-pong)
- Input: previous pass texture, step size (uniform)
- Output: next ping-pong texture (R32F or RG8)
- Logic: JFA step — read 9 texels at offsets (0, ±step, ±step), (0, 0), compute min distance
- Output encodes (distance, seed_x, seed_y) or just distance

### mask_final.frag (1 pass)
- Input: JFA output texture, rect uniforms
- Output: RGBA8 at full res (R=distance, GB=to_center, A=255)
- Logic: sqrt distance, normalize by max, compute direction from nearest rect center

## Rust changes (blur.rs)

- `Blur` struct: add `jfa_textures: [Option<GlesTexture>; 2]` for JFA ping-pong
- `render_custom()`:
  1. Check cache (rects + size unchanged → skip)
  2. Detect overlaps (N*(N-1)/2 rect intersection checks)
  3. Route to cheap or JFA path
  4. Each path: bind FBO to mask texture, set uniforms, draw quad(s)
  5. Bind resulting mask texture as `niri_mask` (unit 1)
  6. Existing down/up/refract/glass passes run unchanged

## Performance

- 3440×1440 full res: cheap path ~0.05ms, JFA path ~1.5ms
- Both within frame budget at 170Hz
- No half-res artifacts
- Cache eliminates cost when nothing changes

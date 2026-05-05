# Mask Distance Field via Euclidean Distance Transform

## Problem

The current mask generation (`generate_mask_data` in `blur.rs`) fills subregion rects with 1.0, then applies a separable box blur. This has two fundamental issues:

1. **Visible seams** — the box blur is a low-quality approximation that produces inconsistent gradients, especially at corners and complex rect boundaries.
2. **No edges for full-window regions** — the blur only creates gradients at the boundary between filled (1.0) and unfilled (0.0) pixels. For a full-window region, every pixel is 1.0, so there is no boundary and no gradient.

Both the refraction shader (`refract.frag`) and glass shader (`glass.frag`) rely on `dFdx(mask)/dFdy(mask)` for edge normals and `1 - mask` for edge proximity. They need a smooth, accurate distance field.

## Approach

Replace the binary-fill-then-blur with a proper **Euclidean Distance Transform** (Felzenszwalb & Huttenlocher algorithm). The EDT computes exact Euclidean distances in O(w×h) time — actually faster than the current O(w×h×kernel) box blur.

### Algorithm

1. Create binary mask at half resolution (`mask_w × mask_h`):
   - If `subregion_rects` is non-empty: pixel = 1.0 if inside any rect, 0.0 otherwise
   - If `subregion_rects` is empty (full-window blur): all pixels = 1.0
2. Force a 1-pixel frame of zeros around the mask edges (row 0, row h-1, col 0, col w-1). This provides boundary conditions for the full-window case (distance to window edge) without affecting subregion correctness (rect boundaries are closer to interior pixels).
3. Run Felzenszwalb & Huttenlocher EDT: two separable passes (rows then columns) computing squared Euclidean distance to nearest 0-pixel.
4. Take `sqrt` of each value.
5. Normalize by dividing by max distance → result in [0, 1].
6. Convert to RGBA8 (`R=byte, G=byte, B=byte, A=255`) and upload as texture.

### Why the frame trick works

- **Full-window (no subregions):** all interior pixels = 1.0, frame = 0.0. EDT gives distance from each pixel to nearest window edge. Center pixel gets highest value.
- **Subregions:** pixels outside rects = 0.0 (naturally). Frame = 0.0 (redundant at edges). EDT gives distance from interior pixels to nearest rect boundary. The frame is farther than rect edges for interior pixels, so it doesn't interfere.
- **Subregion touching window edge:** pixel at window edge has distance 0 from both the rect edge and the frame — correct, since it's at a boundary.

### Performance

- EDT is O(w×h), vs current box blur O(w×h×kernel_size) where kernel_size = 11.
- At half resolution (960×540 for 1080p): ~518K pixels, processed in microseconds.
- Same CPU→GPU upload pattern, no pipeline changes.

## Scope

- Replace `generate_mask_data` function in `src/render_helpers/blur.rs`.
- No shader changes needed — existing shaders read `texture(niri_mask, uv).r` and compute gradients via `dFdx/dFdy`.
- No changes to `framebuffer_effect.rs` or `background_effect.rs`.
- Remove `MASK_BLUR_RADIUS` constant (no longer needed).

## Files changed

- `src/render_helpers/blur.rs`: rewrite `generate_mask_data` to use EDT, remove `MASK_BLUR_RADIUS`.

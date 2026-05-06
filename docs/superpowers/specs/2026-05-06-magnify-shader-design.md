# Magnify Custom Blur Shader -- Design

## Overview

A new custom blur shader pipeline (`shaders/magnify/`) that creates a magnifying-glass
lens effect on background windows. It operates on sharp (unblurred) background content,
producing a fishbowl-style dome distortion: center pixels are magnified (sampled from a
wider UV range), and rim pixels are compressed, simulating the optical profile of a convex
lens. Chromatic aberration separates RGB channels slightly at the edges, as real glass does.

This replaces the 4-pass overshifted3 pipeline with a lean 2-pass pipeline -- no
downsample/upsample blur stages, just the distortion pass followed by a glass highlight
pass.

## Pipeline Structure

```
pass "magnify" file="magnify.frag" scale=1.0
pass "glass"   file="glass.frag"   scale=1.0
```

Both passes run at native resolution (scale=1.0). The custom blur system chains them:
pass 0 output feeds pass 1 input. The GPU mask (`niri_mask`) is pre-rendered by the mask
shader and bound to both passes.

## Files

| File | Purpose |
|------|---------|
| `shaders/magnify/pipeline.kdl` | 2-pass pipeline definition |
| `shaders/magnify/magnify.frag` | Dome lens distortion with chromatic aberration |
| `shaders/magnify/glass.frag` | Glass highlight glow + noise |

## magnify.frag -- Lens Distortion

### Available Uniforms

Same as any custom blur pass:
- `sampler2D niri_input` -- previous pass output (sharp background)
- `sampler2D niri_mask` -- GPU mask (R: intensity, GB: direction to center)
- `vec2 niri_output_size`, `vec2 niri_input_size` -- texture dimensions
- `vec2 niri_half_pixel` -- half-pixel offset for precise sampling

### Core Math: Dome Curve

The mask's R channel (`m = mask_sample.r`) encodes intensity: 0 = outside/no effect,
1 = fully inside at center. The mask GB channels encode `to_center = (gb - 0.5) * 2.0`,
the vector from the current pixel toward the subregion center.

The dome curve transforms this into a signed displacement:

```
normalized = m * 2.0 - 1.0          // range [-1, 1], negative at rim, positive at center
dome = normalized^3 * strength       // cubic S-curve, smooth zero-crossing at m=0.5
```

**Magnitude profile:**
- `m=1.0` (center): `dome = +strength` → push UV away from center → magnification
- `m=0.5` (mid): `dome = 0` → neutral, no displacement
- `m=0.0` (rim): `dome = -strength` → pull UV toward center → edge compression

`strength` defaults to 0.15 and can be tuned. The cubic provides a smooth transition
with stronger curvature near the center (the "domed" feel) and a tapering compression
at the rim.

### UV Displacement

UVs are displaced radially, away from center for magnification, toward center for
compression:

```
// to_center points from UV toward center, so -to_center points away from center
displacement = -to_center * dome

uv_g = uv + displacement
```

### Chromatic Aberration

Real glass lenses refract different wavelengths by different amounts. Blue (shorter
wavelength) bends more; red (longer wavelength) bends less. We apply this as a per-channel
scale on the dome displacement:

```
chromatic = 0.06
uv_r = uv + displacement * (1.0 - chromatic)  // red: less displaced
uv_g = uv + displacement                       // green: base displacement
uv_b = uv + displacement * (1.0 + chromatic)  // blue: more displaced
```

All UVs clamped to [0, 1] after displacement. Final output combines RGB from the three
samples, alpha from the green channel sample.

### Outside Mask

When `mask < 0.001`, pass `niri_input` through unchanged -- no distortion outside the
subregion.

## glass.frag -- Highlights

Nearly identical to `shaders/overshifted3/glass.frag`. The only change is that the mask
profile is different (no blur, sharper falloff), so the `smoothstep(0.2, 0.0, mask)` glow
weighting may benefit from tuning.

Key behaviors:
- Outside mask: passthrough
- Inside mask: compute directional glow from `atan(to_center)` angle
- Asymmetric sine-cubed glow creates a highlight on one side
- Weighted by `smoothstep(0.2, 0.0, mask)` so glow intensifies near edge
- Add subtle random noise (uniform `±0.005` around color channels)

## Configuration

```kdl
blur {
    custom-shader "shaders/magnify"
}
```

No build step required -- shaders load at runtime, same as all custom blur pipelines.

## Constraints & Edge Cases

- **No blur**: Unlike other custom blur shaders, there's no down/up pass pair. The
  background is sharp. This is by design -- the effect is a lens, not frosted glass.
- **Single subregion**: Works best when the mask covers one focused area (the subregion
  around a window). Multiple disjoint subregions would each get their own fishbowl.
- **Corner radius interaction**: The mask's corner SDF affects the mask intensity at
  corners, which means corners of the subregion will have weaker distortion -- consistent
  with how a round lens would behave.
- **UV clamping**: Displaced UVs are clamped to [0,1] to avoid sampling outside texture
  bounds. At extreme displacements this may cause edge smearing, which is acceptable.

# Magnify Shader Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a 2-pass magnifying-glass custom blur shader pipeline that produces fishbowl lens distortion with chromatic aberration on sharp backgrounds.

**Architecture:** Three files in `shaders/magnify/` -- a KDL pipeline definition and two GLSL ES 3.00 fragment shaders. The magnify pass computes a cubic dome displacement curve from the GPU mask, pushing UVs outward for center magnification and inward for rim compression. The glass pass adds directional highlight glow and noise (copied from overshifted3).

**Tech Stack:** GLSL ES 3.00, KDL config format, niri custom blur pipeline runtime

---

### File Structure

```
shaders/magnify/
  pipeline.kdl      -- 2-pass definition (magnify, glass)
  magnify.frag      -- dome lens distortion with chromatic aberration
  glass.frag        -- highlight glow + noise (copy of overshifted3/glass.frag)
```

---

### Task 1: Create pipeline.kdl

**Files:**
- Create: `shaders/magnify/pipeline.kdl`

- [ ] **Step 1: Create the magnify shader directory and pipeline.kdl**

```bash
mkdir -p shaders/magnify
```

- [ ] **Step 2: Write pipeline.kdl**

Write `shaders/magnify/pipeline.kdl`:

```kdl
pass "magnify" file="magnify.frag" scale=1.0
pass "glass" file="glass.frag" scale=1.0
```

Both passes at scale=1.0 (native resolution, no down/up sampling). The custom blur system chains pass 0 output into pass 1 input automatically. The mask texture is bound to any pass that declares `uniform sampler2D niri_mask`.

- [ ] **Step 3: Verify the file**

```bash
cat shaders/magnify/pipeline.kdl
```

Expected: shows the two pass lines above.

- [ ] **Step 4: Commit**

```bash
git add shaders/magnify/pipeline.kdl
git commit -m "feat: add magnify custom blur pipeline definition"
```

---

### Task 2: Create magnify.frag (core lens distortion)

**Files:**
- Create: `shaders/magnify/magnify.frag`

- [ ] **Step 1: Write magnify.frag**

Write `shaders/magnify/magnify.frag`:

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

const float u_magnify_strength = 0.15;
const float u_chromatic = 0.06;

void main() {
    vec2 uv = v_coords;

    vec4 mask_sample = texture(niri_mask, uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;

    float normalized = mask * 2.0 - 1.0;
    float dome = normalized * normalized * normalized * u_magnify_strength;

    vec2 displacement = -to_center * dome;

    vec2 uv_r = clamp(uv + displacement * (1.0 - u_chromatic), 0.0, 1.0);
    vec2 uv_g = clamp(uv + displacement, 0.0, 1.0);
    vec2 uv_b = clamp(uv + displacement * (1.0 + u_chromatic), 0.0, 1.0);

    float cr = texture(niri_input, uv_r).r;
    float cg = texture(niri_input, uv_g).g;
    float cb = texture(niri_input, uv_b).b;
    float ca = texture(niri_input, uv_g).a;

    frag_color = vec4(cr, cg, cb, ca);
}
```

**What this does:**
1. Reads mask R channel (intensity) and GB channels (direction to center)
2. Outside mask (R < 0.001): passthrough, no effect
3. Inside mask: maps `mask` through a cubic S-curve (`normalized = mask*2-1`, then `dome = normalized^3 * strength`)
4. At mask=1 (center): dome positive → displacement points away from center → magnification
5. At mask=0 (rim): dome negative → displacement points toward center → edge compression
6. Applies chromatic aberration: red displaced ~6% less, blue displaced ~6% more than green
7. Clamps all UVs to [0,1] to stay within texture bounds

The cubic (`normalized^3`) gives a smooth zero-crossing at mask=0.5 and strong curvature near center (the "domed" feel).

- [ ] **Step 2: Verify the file syntax (basic check)**

```bash
head -1 shaders/magnify/magnify.frag
```

Expected: `#version 300 es`

- [ ] **Step 3: Commit**

```bash
git add shaders/magnify/magnify.frag
git commit -m "feat: add magnify lens distortion shader pass"
```

---

### Task 3: Create glass.frag (highlight pass)

**Files:**
- Create: `shaders/magnify/glass.frag`

- [ ] **Step 1: Write glass.frag**

Write `shaders/magnify/glass.frag` (copy of overshifted3/glass.frag with mask edge threshold tuned for sharp background):

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;

out vec4 frag_color;

const float u_noise = 0.01;

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = v_coords;

    vec4 mask_sample = texture(niri_mask, uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;
    float dist_center = length(to_center);

    float angle = dist_center > 1e-6 ? atan(to_center.y, to_center.x) : 0.0;
    float glow = pow(max(sin(angle - 0.5), 0.0), 3.0) - pow(max(-sin(angle - 0.5), 0.0), 3.0);
    float glow_weight = 0.6;
    float mul = glow * glow_weight * smoothstep(0.2, 0.0, mask) + 1.0;

    vec4 noise = vec4(vec3(rand(gl_FragCoord.xy) - 0.5), 0.0);
    vec4 color = texture(niri_input, uv) + noise * u_noise;

    color.rgb *= mul;

    frag_color = color;
}
```

This is a copy of `shaders/overshifted3/glass.frag` with one change: the `rand()` call uses `gl_FragCoord.xy` directly (not `gl_FragCoord.xy * 1e-3`) to produce per-pixel noise since we're at native resolution with no blur scaling.

- [ ] **Step 2: Verify the file syntax (basic check)**

```bash
head -1 shaders/magnify/glass.frag
```

Expected: `#version 300 es`

- [ ] **Step 3: Commit**

```bash
git add shaders/magnify/glass.frag
git commit -m "feat: add magnify glass highlight shader pass"
```

---

### Task 4: End-to-End Verification

**Files:**
- Modify: `config.kdl` (temporary, for testing)

- [ ] **Step 1: Verify all three files are present**

```bash
ls -la shaders/magnify/
```

Expected: shows `pipeline.kdl`, `magnify.frag`, `glass.frag`

- [ ] **Step 2: Verify niri builds (unchanged, just a sanity check)**

```bash
cargo build --release 2>&1 | tail -5
```

Expected: successful build. No changes to Rust code, so this should trivially pass.

- [ ] **Step 3: Configure niri to use the shader**

Update `config.kdl` -- change the `custom-shader` line inside the `blur` block:

```kdl
blur {
    custom-shader "shaders/magnify"
}
```

- [ ] **Step 4: Visual verification instructions**

Start niri. The magnify effect should be visible on windows that have blur subregions (e.g., terminal, any app with a semi-transparent background behind blur):

1. **Center magnification**: Content behind the glass should appear enlarged/distorted in a dome pattern
2. **Edge compression**: Near the edges of the blur region, content should be slightly compressed
3. **Chromatic aberration**: Subtle red/blue color fringing visible at high-contrast edges within the lens area
4. **Glass highlight**: A directional highlight glow should be visible on the glass surface
5. **No GLSL errors**: Check niri logs -- no shader compilation errors

If the visual effect is inverted (things look smaller instead of bigger at center), the `dome` sign needs flipping: change `-to_center * dome` to `to_center * dome` in `magnify.frag`.

- [ ] **Step 5: Final commit (if any tuning was done)**

```bash
git add -A
git commit -m "feat: complete magnify custom blur shader pipeline"
```

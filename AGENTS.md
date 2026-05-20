# AGENTS.md

Branch: `feat/custom-blur-shader` — custom shader pipelines for background effects, replacing the old hardcoded Kawase blur.

## What we're building

Custom shader pipelines let users (or more commonly, shader authors) supply their own multi-pass fragment shader programs that run in place of the built-in Kawase blur. Each pipeline is a directory containing a `pipeline.kdl` manifest and `.frag` files. The compositor loads them on demand and caches compiled programs.

This is layered on top of a GPU mask system that tells each shader pass which pixels are "inside" the window (vs. transparent background). Two strategies produce these masks.

## Mask strategies

### 1. Analytical SDF (`mask.frag`)

**When:** The window has zero subregion rects, or a single rect covering the full surface.

**How:** `sdRoundedRect()` computes exact signed-distance to a rounded rectangle. The direction field is just `(0.5, 0.5) - uv` (from center to pixel). Single-pass, no JFA, no multigrid.

**Key files:**
- `src/render_helpers/shaders/mask.frag` — the SDF shader
- `src/render_helpers/blur.rs` — `MaskProgram` (line 245), `compile_mask_program()`, mask render logic (lines 960–1033)

### 2. JFA + Multigrid Poisson (shape-agnostic)

**When:** Multiple subregion rects (explicit blur regions from `ext-background-effect` Wayland protocol).

**How:** A multi-stage GPU pipeline:

1. **Binary mask** (`mask_binary.frag`) — renders the union of subregion rects with anti-aliased coverage. Reads rects from a 1×N RGBA32F texture (unbounded count).
2. **JFA** (Jump Flooding Algorithm) — `jfa_init.frag` seeds exterior pixels, then `jfa_step.frag` iterates at decreasing step sizes to find the nearest exterior pixel for every pixel.
3. **SDF bake** (`jfa_sdf_bake.frag`) — converts JFA nearest-point coordinates to a scalar distance field.
4. **Multigrid Poisson solve** — solves `Δu = -f` (with `f = 1` inside, `0` outside) to produce a smooth direction field:
   - `jfa_poisson_init_rhs.frag` — init level-0 RHS
   - `jfa_poisson_restrict_mask.frag` — box-downsample mask+RHS through 6 levels
   - `jfa_poisson_jacobi.frag` — weighted-Jacobi smoother (damping ω=0.8)
   - `jfa_poisson_residual_restrict.frag` — compute residual, restrict to coarser level
   - `jfa_poisson_prolongate.frag` — bilinear-upsample coarse correction
   - 2 V-cycles with 12 pre/post Jacobi sweeps each
5. **Encode** (`jfa_encode.frag`) — combines JFA distance (R) with Poisson gradient (GB = `∇u` normalized, encoded as `dir*0.5+0.5`)

**Key files:**
- `src/render_helpers/blur.rs` — `JfaPipeline` (line 355), `JfaTextures` (line 382), `MultigridLevel` (line 370), `render_jfa_mask()` (line 1157), `run_v_cycle()` and related functions
- All `src/render_helpers/shaders/jfa_*.frag` and `jfa_poisson_*.frag` files
- `src/render_helpers/shaders/mask_binary.frag` — binary mask from subregion rects

### Bbox caching

The entire JFA pipeline runs in bbox-local coordinates. If the bbox size and per-rect offsets haven't changed between frames (e.g. only cursor movement), the cached `encoded` texture is reused and just re-blitted at the new screen position. See `Blur.cached_jfa_rects_local` and `cached_jfa_bbox_size` fields (blur.rs lines 41–48).

## Key source files

| File | Role |
|------|------|
| `src/render_helpers/blur.rs` (2601 lines) | Central hub: `Blur`, `BlurProgram`, `CustomBlurProgram`, all JFA/Poisson structs and pipeline stages, multigrid V-cycle |
| `src/render_helpers/custom_blur.rs` (95 lines) | `PipelineConfig`, `resolve_pipeline()` reads .frag files from inline config paths, `cache_key()` hashes sources |
| `src/render_helpers/shaders/mod.rs` (440 lines) | `Shaders` struct, compiles built-in shaders, caches custom blur pipelines |
| `src/render_helpers/background_effect.rs` (337 lines) | `BackgroundEffect`, `Options`, `RenderParams` — ties blur config to window/layer rendering |
| `src/render_helpers/framebuffer_effect.rs` (497 lines) | `FramebufferEffect` — captures framebuffer, runs blur, draws result |
| `src/render_helpers/effect_buffer.rs` (325 lines) | `EffectBuffer` — cached offscreen texture + on-demand blur (xray path) |
| `src/render_helpers/xray.rs` (382 lines) | `Xray`, `XrayElement` — transparency rendering |
| `niri-config/src/appearance.rs` | `Blur`, `BlurPart`, `ShaderPipeline`, `MaskPassPart`, `RenderPassPart`, `BackgroundEffect`, `BackgroundEffectRule` config types |
| `src/handlers/background_effect.rs` (124 lines) | Wayland protocol handler for `ext-background-effect` blur regions |
| `src/window/mapped.rs` | Window render calls `background_effect::render_for_tile()` (line 721) |
| `src/layer/mapped.rs` | Layer shell calls `background_effect::render_for_tile()` (line 240) |

## Built-in shader files

All in `src/render_helpers/shaders/`:

- `blur.vert` / `blur_down.frag` / `blur_up.frag` — default Kawase blur
- `blur_custom.vert` — shared vertex shader for all custom/JFA passes
- `mask.frag` — analytical SDF mask
- `mask_binary.frag` — binary subregion mask
- `jfa_init.frag`, `jfa_step.frag`, `jfa_sdf_bake.frag` — JFA stages
- `jfa_poisson_init_rhs.frag`, `jfa_poisson_restrict_mask.frag`, `jfa_poisson_jacobi.frag`, `jfa_poisson_residual_restrict.frag`, `jfa_poisson_prolongate.frag` — multigrid Poisson stages
- `jfa_encode.frag`, `jfa_encode_debug.frag` — final encode
- `border.frag`, `shadow.frag`, `clipped_surface.frag`, `rounding_alpha.frag`, `postprocess.frag` — border/shadow/clipping/post-process

## User-provided pipeline format

A directory containing:
- `pipeline.kdl` — KDL manifest listing render passes and optionally a mask pass:
  ```kdl
  mask-pass "circle_mask" file="circle_mask.frag" scale=1.0
  render-pass "glass" file="glass.frag" scale=1.0
  render-pass "post" file="post.frag" scale=1.0
  ```
- `.frag` files referenced in the manifest

Each **render pass** receives these uniforms: `niri_input` (TEXTURE0), `niri_output_size`, `niri_input_size`, `niri_half_pixel`, `niri_pass`, `niri_pass_count`, `niri_geo_size`, `niri_corner_radius`, `niri_mask` (TEXTURE1, the mask texture), `niri_window_screen_rect` (vec4, window origin and size in screen UV: xy=origin, zw=size).

The optional **mask pass** receives: `niri_subregion_count` (int), `niri_subregion_rects` (sampler2D at TEXTURE1, a 1×N RGBA32F texture of rects in source pixels), `niri_mask_size` (vec2, the render target size), `niri_bbox_origin` (vec2, always (0,0)), `niri_geo_size` (vec2), `niri_corner_radius` (vec4). It outputs to `frag_color` in the same convention as `mask.frag`: R=mask, G=dir_x*0.5+0.5, B=dir_y*0.5+0.5.

When a `mask-pass` is present, the built-in mask heuristic (JFA+Poisson vs. analytical SDF) is bypassed; the custom mask pass runs at full source resolution instead.

See `shaders/` directory for example pipelines (crt, magnify, overshifted3, etc.).

## Rendering flow

```
Window/Layer render loop
  → background_effect::render_for_tile()
    → xray path: EffectBuffer::prepare() → EffectBuffer::render() → Blur::render()
    → non-xray path: FramebufferEffectElement::capture_framebuffer() → Blur::render()

Blur::render() (blur.rs line 2413)
  → if shader_pipeline set: get_or_compile_custom_blur() → Blur::render_custom()
    → if custom mask pass declared: render custom mask pass (at full source resolution)
    → else if subregion_rects non-empty & not single-full: render_jfa_mask() [JFA+Poisson pipeline]
    → else: render analytical SDF mask
    → for each custom render pass: bind textures, set uniforms, draw fullscreen quad
  → else: default Kawase down/up passes
```

## Caching

- **Custom blur programs:** `Shaders::custom_blur` is a `RefCell<HashMap<u64, Option<CustomBlurProgram>>>` — compiled lazily on first use, keyed by a hash of all shader source strings. Failures cached as `None`. Cleared on config reload (`shaders/mod.rs` line 417).
- **JFA output:** The entire pipeline runs in bbox-local coords; caching by bbox size + local rect offsets avoids recomputation when only the cursor/scroll position changes.
- **Mask shader:** `Blur.mask_program` is compiled once per `Blur` instance and reused.
- **Rects texture:** 1×N RGBA32F texture (grows when rect count exceeds capacity, never shrinks).

## Precision notes

- Multigrid `u` textures use raw GL RGBA32F (via `create_rgba32f_buffer`, blur.rs line 410). Half-float is insufficient near gradient minima.
- RHS is scaled by `1/max_dist²` to keep `u` in precision-friendly range (~0.06 peak for RGBA16F RHS textures).
- Bbox calculation has 1e-4 epsilon snap to prevent pixel-coordinate wobble from f32 round-trip error (blur.rs lines 805–818).
- `glColorMask(TRUE, FALSE, FALSE, FALSE)` is used during residual restrict to write only R without clobbering the pre-populated mask in G (blur.rs lines 1990–1992).

## Testing

- **Visual tests:** `niri-visual-tests/` — GTK4/ADW app that renders test cases. Run with `cargo run -p niri-visual-tests`. Test cases live in `niri-visual-tests/src/cases/`.
- **Shader pipeline test directories:** `circle-test/`, `jfa-test/`, `magnify-test/`, `triangle-test/` — each contains a `pipeline.kdl` + fragment shaders for manual visual verification.
- **Config tests:** `insta` snapshot tests in `niri-config/`.
- **Compile check:** `cargo check --all-targets`

## Config integration

```kdl
// Global blur config (pipeline inline, file= paths point to .frag shader files)
blur {
    passes 3
    offset 3.
    noise 0.02
    saturation 1.5
    shader-pipeline {
        render-pass "crt" file="/path/to/crt.frag" scale=1.0
    }
}

// Per-window override
window-rule {
    match app-id="kitty"
    blur { noise 0.5 saturation 2.0 shader-pipeline { render-pass "glass" file="/path/to/glass.frag" scale=1.0 } }
    background-effect {
        xray true
        blur true
    }
}
```

`shader_pipeline` is merged via `merge_clone_opt!` through `Blur` → `BlurPart` and `BackgroundEffect` → `BackgroundEffectRule`. The pipeline is cached in `Shaders::custom_blur` keyed by a hash of all shader source strings.

## Update policy

When anything described in this file changes (new strategies, restructured files, changed caching behavior, new uniforms, changed pipeline format), update this AGENTS.md to reflect the new state. Keep it current for future agents and contributors.

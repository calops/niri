# AGENTS.md

Branch: `feat/custom-blur-shader` — custom shader pipelines for background effects, replacing the old hardcoded Kawase blur.

## Architecture

The blur pipeline runs in three stages:

1. **Initial binary mask** — renders anti-aliased subregion coverage through `mask_binary.frag` when there are no mask passes or the first pass is custom. It is skipped when the first pass is a self-contained built-in vector pass.
2. **Mask passes** — zero or more user-declared passes that refine or replace the mask. Each custom pass reads the previous pass's output. Built-in vector passes are self-contained.
3. **Render passes** — one or more passes that produce the final blurred output. Each pass receives `niri_input` (source texture or previous pass output) and `niri_mask` (final mask texture).

## Mask pass types

| Config value | Description |
|-------------|-------------|
| `"window-vectors"` | Analytical SDF for a full-window rounded rectangle. Produces R=normalized interior distance + GB=vector from the fragment to the center. Uses `geo_size` and `corner_radius`. |
| `"region-vectors"` | JFA distance + Poisson direction field for arbitrary subregion layouts. Works for any composite region. Uses `subregion_rects`. Cached by bbox-local coordinates. |
| `"custom"` | User-provided `.frag` file. Requires `file=` property. Receives the previous mask, or the binary mask when it is the first pass. |

## Render pass types

| Config value | Description |
|-------------|-------------|
| `"dual-kawase-blur"` | Built-in Kawase down+up blur. Inherits `passes` and `offset` from the parent `blur` block. The final mask remains bound, but the current built-in shaders do not sample it. Outputs at source resolution. |
| `"custom"` | User-provided `.frag` file. Requires `file=` property. Receives `niri_input`, `niri_mask`, and geometry uniforms. |

## Mask texture convention

The binary mask is an intermediate coverage texture: R is anti-aliased coverage and GBA are zero.

The built-in vector masks use:
- **R** = normalized interior distance
- **G** = direction_x * 0.5 + 0.5
- **B** = direction_y * 0.5 + 0.5

`window-vectors` encodes the vector to the window center. `region-vectors` encodes the scaled inward Poisson-gradient direction. Custom passes receive the previous texture and must output the convention expected by their render passes.

## Key source files

| File | Role |
|------|------|
| `src/render_helpers/blur.rs` (~2970 lines) | Central hub: `Blur`, `BlurProgram`, `CustomBlurProgram`, `MaskProgram` (analytical SDF), `JfaPipeline`/`JfaTextures`/`MultigridLevel` (region-vectors), final-mask and bbox-local JFA caches, `render_custom()`, `render_jfa_mask()`, `run_dual_kawase_blur()`, and multigrid V-cycle functions. Constants: `MAX_KAWASE_PASSES`, `JFA_CACHE_EPSILON`, `MAX_MULTIGRID_LEVELS`, `MULTIGRID_COARSE_SIZE`, `JACOBI_OMEGA`, `JACOBI_SWEEPS`, `V_CYCLES`, `BBOX_BORDER`. |
| `src/render_helpers/custom_blur.rs` (140 lines) | `MaskPassStep`/`RenderPassStep` enums, `PipelineConfig`, `resolve_pipeline()`, `cache_key()` |
| `src/render_helpers/shaders/mod.rs` (450 lines) | `Shaders` struct, compiles built-in shaders, caches custom blur pipelines |
| `niri-config/src/appearance.rs` | `Blur`, `BlurPart`, `ShaderPipeline`, `MaskPassPart`/`RenderPassPart` (with `kind` discriminant: `MaskPassKind`/`RenderPassKind`), `BackgroundEffect`, `BackgroundEffectRule` |
| `src/render_helpers/background_effect.rs` | `BackgroundEffect`, `Options`, `RenderParams` — ties blur config to window/layer rendering |
| `src/render_helpers/framebuffer_effect.rs` | `FramebufferEffect` — captures framebuffer, runs blur, draws result |
| `src/render_helpers/effect_buffer.rs` | `EffectBuffer` — cached offscreen texture + on-demand blur (xray path) |
| `src/render_helpers/xray.rs` | `Xray`, `XrayElement` — transparency rendering |
| `src/handlers/background_effect.rs` | Wayland protocol handler for `ext-background-effect` blur regions |
| `src/window/mapped.rs` | Window render calls `background_effect::render_for_tile()` |
| `src/layer/mapped.rs` | Layer shell calls `background_effect::render_for_tile()` |

## Built-in shader files

All in `src/render_helpers/shaders/`:

- `blur.vert` / `blur_down.frag` / `blur_up.frag` — default Kawase blur (also used by `dual-kawase-blur`)
- `blur_custom.vert` — shared vertex shader for all custom/JFA/binary passes
- `mask.frag` — analytical SDF mask (used by `window-vectors`)
- `mask_binary.frag` — binary subregion coverage mask
- `jfa_init.frag`, `jfa_step.frag` — JFA stages used by `region-vectors`
- `jfa_poisson_init_rhs.frag`, `jfa_poisson_restrict_mask.frag`, `jfa_poisson_jacobi.frag`, `jfa_poisson_residual_restrict.frag`, `jfa_poisson_prolongate.frag` — multigrid Poisson stages
- `jfa_encode.frag`, `jfa_smooth.frag` — direction encode and cached 3×3 tent smoothing for JFA+Poisson
- `border.frag`, `shadow.frag`, `clipped_surface.frag`, `rounding_alpha.frag`, `postprocess.frag` — border/shadow/clipping/post-process

## Config format

```kdl
blur {
    passes 3
    offset 3.
    noise 0.02
    saturation 1.5
    shader-pipeline {
        mask-pass "window-vectors"
        mask-pass "custom" file="/path/to/refine.frag" scale=1.0
        render-pass "dual-kawase-blur"
        render-pass "custom" file="/path/to/glass.frag" scale=1.0
    }
}

window-rule {
    match app-id="kitty"
    background-effect {
        blur true
        shader-pipeline {
            mask-pass "region-vectors"
            render-pass "dual-kawase-blur"
        }
    }
}

layer-rule {
    match namespace="quickshell"
    background-effect {
        blur true
        shader-pipeline {
            mask-pass "region-vectors"
            render-pass "custom" file="/path/to/blur.frag" scale=1.0
        }
    }
}
```

`shader_pipeline` is merged via `merge_clone_opt!` through `Blur` → `BlurPart` and `BackgroundEffect` → `BackgroundEffectRule`. The pipeline is cached in `Shaders::custom_blur` keyed by a hash of all shader source strings and pipeline scales.

## Uniforms

### Custom mask pass (`CustomMaskProgram`)

| Uniform | Type | Slot | Description |
|---------|------|------|-------------|
| `niri_subregion_count` | int | — | Number of subregion rects |
| `niri_subregion_rects` | sampler2D | TEXTURE1 | 1×N RGBA32F texture of rects in source pixels |
| `niri_output_size` | vec2 | — | Render target size |
| `niri_bbox_origin` | vec2 | — | Always (0,0) |
| `niri_geo_size` | vec2 | — | Window geometry size |
| `niri_corner_radius` | vec4 | — | Corner radii |

Custom mask passes receive `niri_mask` (the previous mask) implicitly via TEXTURE0. A first custom pass receives the binary coverage mask; a custom pass after another pass receives that pass's output.

### Custom render pass (`CustomBlurPassProgram`)

| Uniform | Type | Slot | Description |
|---------|------|------|-------------|
| `niri_input` | sampler2D | TEXTURE0 | Source texture or previous pass output |
| `niri_output_size` | vec2 | — | Output texture size |
| `niri_input_size` | vec2 | — | Input texture size |
| `niri_half_pixel` | vec2 | — | Half pixel for sampling |
| `niri_pass` | int | — | 0-based pass index |
| `niri_pass_count` | int | — | Total render pass count |
| `niri_geo_size` | vec2 | — | Window geometry size |
| `niri_corner_radius` | vec4 | — | Corner radii |
| `niri_mask` | sampler2D | TEXTURE1 | Final mask texture |
| `niri_window_screen_rect` | vec4 | — | Window origin (xy) and size (zw) in screen UV |

## Rendering flow

```
Renderer/Layer render loop
  → background_effect::render_for_tile()
    → xray path: EffectBuffer::prepare() → EffectBuffer::render() → Blur::render()
    → non-xray path: FramebufferEffectElement::capture_framebuffer() → Blur::render()
    → (both paths pass BlurOptions including shader_pipeline)

Blur::render()
  → if shader_pipeline set: get_or_compile_custom_blur() → Blur::render_custom()
    → if final-mask cache matches: reuse the active mask texture
    → otherwise:
        - render binary mask only when there are no mask passes or the first pass is custom
        - run each mask pass:
            - "window-vectors": analytical SDF → dst
            - "region-vectors": JFA+Poisson pipeline → dst
            - "custom": user shader (reads src, writes dst ping-pong)
    → run each render pass:
        - "dual-kawase-blur": run_dual_kawase_blur() (down+up chain)
        - "custom": user shader with niri_mask bound
  → else: default Kawase down/up passes
```

Mask ping-pong: `mask_texture_a` and `mask_texture_b` alternate as source/destination across mask passes. Built-in passes overwrite their destination and do not read their source.

## Caching

- **Custom blur programs:** `Shaders::custom_blur` is a `RefCell<HashMap<u64, Option<CustomBlurProgram>>>` — compiled lazily on first use. Failures are cached as `None`. Cleared on config reload.
- **Final mask:** cached by compiled pipeline identity, source size, subregion rectangles, geometry size, and corner radii. Exact hits skip the entire binary and mask pipeline.
- **JFA output:** computed in bbox-local coordinates and cached by bbox size + local rect offsets. A translation cache hit re-blits the encoded texture without recomputing JFA/Poisson. When `region-vectors` is the first pass and reuses the same destination texture, only the previous bbox is cleared.
- **Mask programs:** `Blur.mask_program`, `Blur.binary_program`, and `Blur.jfa_pipeline` are compiled once per `Blur` instance and reused.
- **Rects texture:** 1×N RGBA32F texture (grows when rect count exceeds capacity, never shrinks). Pixel-space rectangle conversion and upload are reused across mask paths.

## Error handling

- Invalid `kind` values are rejected at config parse time (knuffel enum decode).
- Missing `file=` on custom passes is rejected in `resolve_pipeline()` with a clear error.
- Invalid shader files produce warnings and the pipeline falls through to the default Kawase blur.
- Compilation failures are cached as `None` to avoid retry-storms.

## Testing

- **Visual tests:** `niri-visual-tests/` — GTK4/ADW app. `cargo run -p niri-visual-tests`
- **Shader pipeline examples:** `shaders/crt/`, `shaders/magnify/`, `shaders/overshifted3/`, `shaders/debug-mask/`, `shaders/debug-edges/`, `shaders/jfa-debug/`
- **Config tests:** `insta` snapshot tests in `niri-config/`. Run with `cargo test -p niri-config`
- **Compile check:** `cargo check --all-targets`
- **Nested refresh:** the Winit backend reads the host monitor refresh when available. Set `NIRI_WINIT_REFRESH_RATE=<hz>` to override unreliable Wayland monitor metadata, for example `NIRI_WINIT_REFRESH_RATE=170 target/debug/niri --config config.kdl`.
- **Cava refresh:** `cava-test/shell.qml` follows the nested output refresh by default. Set `NIRI_CAVA_FRAMERATE=<fps>` to hold Cava at a separate rate when isolating compositor timing from region-update frequency.

## Precision notes

- Multigrid `u` textures use raw GL R32F. Half-float is insufficient near gradient minima.
- Multigrid RHS uses RG16F and bbox-local binary coverage uses R16F.
- RHS is scaled by `1/max_dist²` to keep `u` in a precision-friendly range.
- The multigrid pyramid grows dynamically until both coarse dimensions are at most 8 pixels, capped at 12 levels. The solver runs one V-cycle with three pre- and three post-Jacobi sweeps.
- The region-vector encoder takes a central-difference Poisson gradient and fades it with `smoothstep(0.02, 0.10, gmag * max_dist)`. A following cached pass applies a 3×3 tent filter to normalized GB direction only, using the unused JFA ping-pong texture as encode scratch; the JFA-derived R distance channel remains exact.
- Bbox calculation has a 1e-4 epsilon snap to prevent pixel-coordinate wobble.
- `glColorMask(TRUE, FALSE, FALSE, FALSE)` is used during residual restrict to write only R.

## Update policy

When anything described in this file changes, update this AGENTS.md to reflect the new state.

# AGENTS.md

Branch: `feat/custom-blur-shader` — custom shader pipelines for background effects, replacing the old hardcoded Kawase blur.

## Architecture

The blur pipeline runs in three stages:

1. **Initial binary mask** — renders anti-aliased subregion coverage through the instanced `mask_binary.vert` / `mask_binary.frag` pair when there are no mask passes or the first pass is custom, or cursor-coverage coverage through `mask_coverage.frag` when a coverage source is set and the first pass is custom. It is skipped when the first pass is a self-contained built-in vector pass.
2. **Mask passes** — zero or more user-declared passes that refine or replace the mask. Each custom pass reads the previous pass's output. Built-in vector passes are self-contained.
3. **Render passes** — one or more passes that produce the final blurred output. Each pass receives `niri_input` (source texture or previous pass output) and `niri_mask` (final mask texture).

## Mask pass types

| Config value | Description |
|-------------|-------------|
| `"window-vectors"` | Analytical SDF for a full-window rounded rectangle. Produces R=normalized interior distance + GB=vector from the fragment to the center. Uses `geo_size` and `corner_radius`. |
| `"region-vectors"` | JFA distance + Poisson direction field for arbitrary subregion layouts. Works for any composite region. Uses `subregion_rects`. Cached by bbox-local coordinates. |
| `"cursor-vectors"` | Same JFA + Poisson pipeline as `region-vectors`, but seeded from the cursor alpha silhouette instead of subregion rects. Uses `niri_coverage` (premultiplied cursor alpha) and an explicit silhouette bbox. |
| `"custom"` | User-provided `.frag` file. Requires `file=` property. Receives the previous mask, or the coverage mask when it is the first pass of a cursor pipeline. Custom mask passes are not eligible for the exact GPU output clip, so their render shaders must write transparent output outside the shape. |

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

`window-vectors` encodes the vector to the window center. `region-vectors` encodes the scaled inward Poisson-gradient direction. `cursor-vectors` produces the same field for the cursor silhouette. Custom passes receive the previous texture and must output the convention expected by their render passes.

## Key source files

| File | Role |
|------|------|
| `src/render_helpers/blur.rs` (~3500 lines) | Central hub: `Blur`, `BlurProgram`, `CustomBlurProgram`, `MaskProgram` (analytical SDF), `JfaPipeline`/`JfaTextures`/`MultigridLevel` (region- and cursor-vectors), `BlurOptions`/`CoverageMask` (mask source: subregion rects or cursor coverage), final-mask and bbox-local JFA caches, exact output-mask compositing, `render_custom()`, `render_jfa_mask()`, `render_coverage_mask()`, `run_dual_kawase_blur()`, and multigrid V-cycle functions. Constants: `MAX_KAWASE_PASSES`, `JFA_CACHE_EPSILON`, `MAX_MULTIGRID_LEVELS`, `MULTIGRID_COARSE_SIZE`, `JACOBI_OMEGA`, `JACOBI_SWEEPS`, `V_CYCLES`, `BBOX_BORDER`. |
| `src/render_helpers/custom_blur.rs` (140 lines) | `MaskPassStep`/`RenderPassStep` enums, `PipelineConfig`, `resolve_pipeline()`, `cache_key()` |
| `src/render_helpers/cursor_effect.rs` | `CursorEffect`, `CursorEffectGeometry`, `MotionState`, `cursor_effect_geometry()`, `cursor_frame_identity()` — renders the cursor as a padded `FramebufferEffectElement` through the shader pipeline and tracks per-output Clock-based velocity deformation. |
| `src/cursor.rs` | `CursorManager`, `CursorTextureCache` (plain cursor buffers and per-frame alpha coverage textures for the shader mask). |
| `src/render_helpers/shaders/mod.rs` (450 lines) | `Shaders` struct, compiles built-in shaders, caches custom blur pipelines |
| `niri-config/src/appearance.rs` | `Blur`, `BlurPart`, `ShaderPipeline`, `MaskPassPart`/`RenderPassPart` (with `kind` discriminant: `MaskPassKind`/`RenderPassKind`, including `CursorVectors`), `BackgroundEffect`, `BackgroundEffectRule` |
| `niri-config/src/misc.rs` | `Cursor`/`CursorPart` — xcursor theme/size, `effect-padding`, `motion-effect-strength`, and the cursor `shader_pipeline` |
| `src/render_helpers/background_effect.rs` | `BackgroundEffect`, `Options`, `RenderParams` — ties blur config to window/layer rendering |
| `src/render_helpers/framebuffer_effect.rs` | `FramebufferEffect` — captures the visible framebuffer crop, keeps subregion masks in stable full-surface coordinates, supplies the visible mask UV crop to render passes, runs blur, and draws exact GPU-masked outputs without expanding damage into one scissor per region rectangle |
| `src/render_helpers/effect_buffer.rs` | `EffectBuffer` — cached offscreen texture + on-demand blur (xray path) |
| `src/render_helpers/xray.rs` | `Xray`, `XrayElement` — transparency rendering |
| `src/handlers/background_effect.rs` | Wayland protocol handler for `ext-background-effect` blur regions |
| `src/window/mapped.rs` | Window render calls `background_effect::render_for_tile()` |
| `src/layer/mapped.rs` | Layer shell calls `background_effect::render_for_tile()` |

## Built-in shader files

All in `src/render_helpers/shaders/`:

- `blur.vert` / `blur_down.frag` / `blur_up.frag` — default Kawase blur (also used by `dual-kawase-blur`)
- `blur_custom.vert` — shared vertex shader for custom, JFA, analytical-mask, and output-mask fullscreen passes
- `mask.frag` — analytical SDF mask (used by `window-vectors`)
- `mask_binary.vert` / `mask_binary.frag` — binary subregion coverage mask; one instanced quad per rectangle with `GL_MAX` coverage blending, used both source-sized and bbox-local for JFA
- `mask_coverage.frag` — writes cursor alpha or hotspot-anchored velocity-deformed SDF coverage into the bbox-local binary texture; the seed for `cursor-vectors`
- `jfa_init.frag`, `jfa_step.frag` — JFA stages used by `region-vectors`
- `jfa_poisson_init_rhs.frag`, `jfa_poisson_restrict_mask.frag`, `jfa_poisson_jacobi.frag`, `jfa_poisson_residual_restrict.frag`, `jfa_poisson_prolongate.frag` — multigrid Poisson stages
- `jfa_encode.frag`, `jfa_smooth.frag` — direction encode and cached 3×3 tent smoothing for JFA+Poisson
- `mask_output.frag` — multiplies eligible custom-pipeline output by the exact built-in region mask, making pixels outside the protocol region transparent
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

cursor {
    // Margin around the cursor silhouette that the shader may refract the
    // backdrop from, in logical pixels.
    effect-padding 32
    shader-pipeline {
        mask-pass "cursor-vectors"
        render-pass "custom" file="/path/to/cursor-glass.frag" scale=1.0
    }
}
```

`shader_pipeline` is merged via `merge_clone_opt!` through `Blur` → `BlurPart` and `BackgroundEffect` → `BackgroundEffectRule`. The pipeline is cached in `Shaders::custom_blur` keyed by a hash of all shader source strings and pipeline scales.

## Uniforms

### Custom mask pass (`CustomMaskProgram`)

| Uniform | Type | Slot | Description |
|---------|------|------|-------------|
| `niri_subregion_count` | int | — | Number of subregion rects |
| `niri_subregion_rects` | sampler2D | TEXTURE1 | Row-major RGBA32F texture of rects in source pixels. Texel `i` is at `(i % textureSize(...).x, i / textureSize(...).x)`; textures wider than `GL_MAX_TEXTURE_SIZE` wrap to additional rows. |
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
| `niri_mask_uv_rect` | vec4 | — | Visible input crop in full-mask UV: `(x1, y1, x2, y2)`. Sample the mask with `mix(niri_mask_uv_rect.xy, niri_mask_uv_rect.zw, v_coords)`. |
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
        - render binary mask only when there are no mask passes or the first pass is custom;
          one instanced quad per rectangle replaces the old per-fragment scan of every rectangle
        - run each mask pass:
            - "window-vectors": analytical SDF → dst
            - "region-vectors": instanced bbox-local binary mask → JFA+Poisson pipeline → dst
            - "cursor-vectors": cursor alpha → bbox-local JFA+Poisson pipeline → dst
            - "custom": user shader (reads src, writes dst ping-pong)
    → run each render pass:
        - "dual-kawase-blur": run_dual_kawase_blur() (down+up chain)
        - "custom": user shader with niri_mask bound
    → when the final mask is exact for the protocol region (binary or final `region-vectors`), apply `mask_output.frag`
      so the final texture is transparent outside the region
  → else: default Kawase down/up passes
```

Mask ping-pong: `mask_texture_a` and `mask_texture_b` alternate as source/destination across mask passes. Built-in passes overwrite their destination and do not read their source.

The cursor shader pipeline reuses this machinery from `Niri::render_pointer`. The effect region is the cursor silhouette expanded by `cursor.effect-padding`, so the captured `niri_input` lets refraction sample the backdrop beyond the silhouette; the silhouette mask itself stays tight and is placed in the padded mask at `CursorEffectGeometry::coverage_bbox`. `FramebufferEffect::render` is called per output with a namespaced Id, so the effect cache is per output. The element replaces the plain cursor entirely and is `Kind::Unspecified`, so it can never be promoted to a DRM cursor plane. Client `wl_surface` cursors fall back to the plain cursor, as do quarter-turn output transforms (Normal and Flipped180 are supported; Flipped180 is what the winit backend always uses for the `winit` connector). The cursor effect is only used on the direct output render path; screenshots and screencasts pass `allow_cursor_effect = false` and fall back to the plain cursor, because they relocate or embed pointer elements where the backdrop needed for refraction is not available.

## Caching

- **Custom blur programs:** `Shaders::custom_blur` is a `RefCell<HashMap<u64, Option<CustomBlurProgram>>>` — compiled lazily on first use. Failures are cached as `None`. Cleared on config reload.
- **Final mask:** cached by compiled pipeline identity, full surface mask size, `Arc` identity of normalized subregion rectangles, geometry size, and corner radii. Exact hits skip the entire binary and mask pipeline.
 - **JFA output:** computed in bbox-local full-surface coordinates and cached by bbox size + local rect offsets. A translation cache hit re-blits the encoded texture without recomputing JFA/Poisson. When `region-vectors` is the first pass and reuses the same destination texture, only the previous bbox is cleared.
- **Cursor mask:** the `cursor-vectors` JFA cache additionally keys on `cursor_frame_identity()` (icon, scale, animation frame, size). Cursor motion is tracked per output with the compositor `Clock`; its hotspot-anchored SDF stretch (up to 35%, controlled by `cursor.motion-effect-strength`, with a symmetric 55 ms velocity low-pass for acceleration and deceleration) invalidates the mask while settling, so the JFA+Poisson field follows the deformed silhouette. JFA+Poisson otherwise reruns merely on animation frames, which are recomputed per frame for now (caching per animation frame is a possible optimization). Named cursor shape transitions are tracked independently per output namespace using the compositor `Clock`; their endpoint SDFs are placed in a hotspot-aligned padded union canvas and sampled with independent normalized UVs.
- **Cursor coverage textures:** `CursorTextureCache` imports each named cursor frame's premultiplied ARGB into a `GlesTexture` once per (icon, scale, frame), cleared on cursor config reload. It also caches CPU-built R32F signed-distance rasters (negative inside) per GPU context. Per-output `CursorEffect` transitions named shape changes for `cursor.shape-transition-duration-ms` (default 150) by interpolating SDFs in `mask_coverage.frag` before thresholding; ordinary xcursor animation frame changes do not transition.
- **Exact output clip:** custom pipelines whose final mask is binary or `region-vectors` run one final `mask_output.frag` pass. The pass transforms visible input UV through `niri_mask_uv_rect` before sampling the stable full-surface mask. With zero post-process noise, `FramebufferEffect` draws compositor damage directly instead of expanding it to thousands of subregion scissors. Other masks, fallback pipelines, and noisy output retain CPU-side region clipping.
- **Normalized subregions:** `FramebufferEffect` caches its normalized float rectangle vector by source `Arc`, relative offset, scale, and full surface size. Sliding partially offscreen effects reuse the full mask and change only `niri_mask_uv_rect`, avoiding per-frame JFA/Poisson recomputation.
- **Mask programs:** `Blur.mask_program`, `Blur.binary_program`, `Blur.mask_output_program`, and `Blur.jfa_pipeline` are compiled once per `Blur` instance and reused.
- **Rects texture:** row-major RGBA32F texture (grows when rect count exceeds capacity, never shrinks) whose width is capped at `GL_MAX_TEXTURE_SIZE` and whose excess texels wrap to additional rows. Pixel-space rectangle conversion and upload are reused across mask paths. Binary mask generation fetches one rect per instanced quad, so changing a large region costs roughly its covered fragments rather than mask pixels × rectangle count.

## Error handling

- Invalid `kind` values are rejected at config parse time (knuffel enum decode).
- Missing `file=` on custom passes is rejected in `resolve_pipeline()` with a clear error.
- Invalid shader files produce warnings and the pipeline falls through to the default Kawase blur.
- Compilation failures are cached as `None` to avoid retry-storms.

## Testing

- **Visual tests:** `niri-visual-tests/` — GTK4/ADW app. `cargo run -p niri-visual-tests`
- **Shader pipeline examples:** `shaders/crt/`, `shaders/magnify/`, `shaders/overshifted3/`, `shaders/liquid-glass-faithful/`, `shaders/prism-glass/`, `shaders/cursor-glass/`, `shaders/debug-mask/`, `shaders/debug-edges/`, `shaders/jfa-debug/`. `liquid-glass-faithful` ports OverShifted/LiquidGlass's analytical radial refraction curve, angular glow, blur, and noise to niri's window-vector mask. `prism-glass` is a separate experimental physically motivated bevel stack with window and region variants.
- **Config tests:** `insta` snapshot tests in `niri-config/`. Run with `cargo test -p niri-config`
- **Compile check:** `cargo check --all-targets`
- **Nested refresh:** the Winit backend reads the host monitor refresh when available. Set `NIRI_WINIT_REFRESH_RATE=<hz>` to override unreliable Wayland monitor metadata, for example `NIRI_WINIT_REFRESH_RATE=170 target/debug/niri --config config.kdl`.
- **Cava refresh:** `cava-test/shell.qml` follows the nested output refresh by default. Set `NIRI_CAVA_FRAMERATE=<fps>` to hold Cava at a separate rate when isolating compositor timing from region-update frequency.
- **Shader background:** `shader-bg-test/` renders the shadertoy shader in `truchet.frag` on a fullscreen background layer surface (namespace `shader-background`). `waves.frag` is kept as the previous wallpaper; point `shell.qml`'s `sourcePath` at it to switch back. `config.kdl` spawns it with `qs -n -p $HOME/projects/niri/shader-bg-test`, so a winit niri gets the background inside its own session. `compile-shader.sh` bakes the `.frag` into a `.qsb` (Qt 6 rejects raw GLSL) cached in `$XDG_STATE_HOME/quickshell/shaders`; it prefers `qsb` from PATH and otherwise takes the newest `qtshadertools` in the nix store, because a spawned shell inherits the session PATH rather than a quickshell wrapper's.
- **Manual showcase:** `config.kdl` starts only `shader-bg-test`, and `Mod+Return` runs `showcase/terminal`, a reproducible transparent Kitty rooted at this checkout. Plain Kitty windows tile at one-third output width with pronounced rounded corners, drop shadows but no client decorations or compositor borders, and the global `liquid-glass-faithful` pipeline. From the last remaining terminal, run `showcase/1`, `showcase/2`, `showcase/3`, then `showcase/4`; they launch the aligned blank shape windows, `wl-bad-apple`, a top-right floating mpv Rickroll, and the Cava Quickshell layer respectively. The Cava launcher defaults `NIRI_CAVA_FRAMERATE` to 60. `NIRI_BAD_APPLE_PROJECT`, `NIRI_RICKROLL_SOURCE`, and an explicitly set `NIRI_CAVA_FRAMERATE` override their defaults.
- **Stained-glass lock UI:** `./lock.sh` launches the standalone native Rust client at `${NIRI_GLASS_LOCK_PROJECT:-$HOME/projects/niri-glass-lock}` through its Nix flake. Two top-level half-screen overlay layer surfaces move by layer-shell margins, each with one static SHM-rendered stained-glass buffer and a surface-local background-effect region; frame callbacks pace the motion without rebuilding the Voronoi cells or mask strips. The `stained-glass-lock` layer rule uses the experimental `prism-glass` region shader pipeline. It accepts any password and does not acquire a Wayland session lock. A real lock will require a niri config option to keep rendering normal surfaces beneath the transparent session-lock surface.

## Precision notes

- Multigrid `u` textures use raw GL R32F. Half-float is insufficient near gradient minima.
- Multigrid RHS uses RG16F and bbox-local binary coverage uses R16F.
- RHS is scaled by `1/max_dist²` to keep `u` in a precision-friendly range.
- The multigrid pyramid grows dynamically until both coarse dimensions are at most 8 pixels, capped at 12 levels. The solver runs one V-cycle with three pre- and three post-Jacobi sweeps.
- The region-vector encoder takes a central-difference Poisson gradient and fades it with `smoothstep(0.02, 0.10, gmag * max_dist)`. It uses the bbox-local binary mask as the authoritative interior/exterior classification so half-float JFA seed quantization cannot leave a non-zero R distance outside the region. A following cached pass applies a 3×3 tent filter to normalized GB direction only, using the unused JFA ping-pong texture as encode scratch; the JFA-derived R distance channel remains exact.
- Bbox calculation has a 1e-4 epsilon snap to prevent pixel-coordinate wobble.
- `glColorMask(TRUE, FALSE, FALSE, FALSE)` is used during residual restrict to write only R.

## Update policy

When anything described in this file changes, update this AGENTS.md to reflect the new state.

# AGENTS.md

Branch: `feat/custom-blur-shader` — custom shader pipelines for background effects, replacing the old hardcoded Kawase blur.

## Architecture

The custom pipeline runs in two stages:

1. **Automatic field generation** — niri owns shape classification. A full window uses one analytical rounded-rectangle draw. Pairwise-independent simple region rectangles use one instanced analytical draw. Touching or overlapping rectangles, connected composite layouts, and cursor silhouettes use the bbox-local JFA + Poisson path.
2. **Color stages** — one or more `blur` or `shader` nodes process the captured backdrop. A custom shader reads the source or previous color output from `niri_color` and the compositor-owned geometry field from `niri_field`.

Shape complexity is not configurable. Analytical evaluation is valid only when every output pixel belongs to at most one simple rectangle. Expanded rectangles are sweep-tested before rendering; touching or overlap routes through JFA/Poisson so a connected union gets one composite field rather than one field per contributing rectangle.

## Pipeline nodes

| Config node | Description |
|-------------|-------------|
| `blur` | Built-in dual-Kawase down/up blur. Optional `passes=` and `offset=` override the parent blur settings for this stage. |
| `shader file="file.frag"` | Custom color stage loaded relative to the declaring config file. The legacy positional path remains accepted. |
| `shader inline="..."` | Inline custom color stage. Optional `scale=` changes either custom shader form's output size. |

Every pipeline declares `version=1`. At least one `blur` or `shader` node is required. Shader files are read during config parsing and watched as config dependencies; rendering never performs shader-file I/O.

## Field texture ABI (version 1)

The compositor-owned automatic field uses one stable `RGBA16F` contract:

- **R** = interior distance in logical pixels; zero outside
- **GB** = signed inward unit direction multiplied by confidence, in `[-1, 1]`
- **A** = authoritative anti-aliased coverage

Coverage is independent of distance. Partially covered boundary pixels may have zero distance and non-zero A. The analytical full-window GB field follows the exact rounded-SDF inward normal and fades to zero by `distance == corner_radius`, before nearest-edge Voronoi boundaries begin; this keeps refraction corner-aligned and the interior seam-free. Independent analytical rectangles use rectangle-local center flow. JFA fields use the inward Poisson gradient with gradient and medial confidence.

## Key source files

| File | Role |
|------|------|
| `src/render_helpers/blur.rs` (~3600 lines) | Central hub: automatic field classification, analytical window/instanced-rectangle fields, JFA/Poisson fields, ordered custom color execution, field and JFA caches, exact field-alpha clipping, dual Kawase, and multigrid V-cycles. |
| `src/render_helpers/custom_blur.rs` | `RenderPassStep`, `PipelineConfig`, version/scale validation, and compact decoded-source cache keys. |
| `src/render_helpers/cursor_effect.rs` | `CursorEffect`, `CursorEffectGeometry`, `MotionState`, `cursor_effect_geometry()`, `cursor_frame_identity()` — padded cursor capture and per-output velocity deformation. |
| `src/cursor.rs` | Plain cursor buffers, alpha coverage textures, and cached cursor SDF rasters. |
| `src/render_helpers/shaders/mod.rs` | Built-in shader compilation and custom pipeline cache. |
| `niri-config/src/appearance.rs` | `ShaderPipeline`, ordered color-stage nodes, blur/background-effect configuration, and merge rules. |
| `niri-config/src/shader.rs` | Shared inline-or-file shader source decoding, config-relative file loading, source fingerprints, and watch dependency registration. |
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

- `blur.vert` / `blur_down.frag` / `blur_up.frag` — default and pipeline `blur` stages
- `blur_custom.vert` — shared fullscreen vertex shader for custom, JFA, analytical-window, and output-clip passes
- `mask.frag` — full-window rounded-rectangle field in the version-1 ABI
- `mask_rect_field.vert` / `mask_rect_field.frag` — one instanced analytical field draw for independent simple rectangles
- `mask_binary.vert` / `mask_binary.frag` — instanced AA union coverage used as the JFA seed
- `mask_coverage.frag` — cursor alpha or hotspot-anchored velocity-deformed SDF coverage used as the cursor JFA seed
- `jfa_init.frag`, `jfa_step.frag` — bbox-local nearest-exterior solve for composite regions and cursors
- `jfa_poisson_init_rhs.frag`, `jfa_poisson_restrict_mask.frag`, `jfa_poisson_jacobi.frag`, `jfa_poisson_residual_restrict.frag`, `jfa_poisson_prolongate.frag` — multigrid Poisson stages
- `jfa_encode.frag`, `jfa_smooth.frag` — version-1 field encoding and cached 3×3 GB smoothing
- `mask_output.frag` — multiplies color by authoritative field alpha outside the protocol region
- `border.frag`, `shadow.frag`, `clipped_surface.frag`, `rounding_alpha.frag`, `postprocess.frag` — border/shadow/clipping/post-process

## Config format

```kdl
blur {
    passes 3
    offset 3.
    noise 0.02
    saturation 1.5
    shader-pipeline version=1 {
        blur passes=2 offset=2.
        shader file="shaders/glass.frag"
    }
}

window-rule {
    match app-id="kitty"
    background-effect {
        blur true
        shader-pipeline version=1 {
            blur
            shader inline="#version 300 es\n// ..."
        }
    }
}

cursor {
    effect-padding 32
    shader-pipeline version=1 {
        shader file="shaders/cursor-glass.frag"
    }
}
```

`shader_pipeline` is merged via `merge_clone_opt!` through `Blur` → `BlurPart` and `BackgroundEffect` → `BackgroundEffectRule`. `Shaders::custom_blur` keys compiled pipelines by ABI version, decoded shader-source fingerprints, and stage options. Animation `custom-shader "..."` keeps its upstream positional-inline syntax and additionally accepts `file=`.

Relative shader paths resolve against the config or included config file that declares them. Shader contents are loaded once while parsing the config and retained through `Arc<str>`. Each resolved file is added to the config watch set, so editing it reloads the config and invalidates affected compiled programs.

## Uniforms

### Color shader (`CustomBlurPassProgram`)

| Uniform | Type | Slot | Description |
|---------|------|------|-------------|
| `niri_color` | sampler2D | TEXTURE0 | Captured backdrop or previous color-stage output |
| `niri_output_size` | vec2 | — | Current output texture size |
| `niri_input_size` | vec2 | — | Current color input texture size |
| `niri_half_pixel` | vec2 | — | Half output pixel in UV |
| `niri_pass` | int | — | Zero-based color-stage index |
| `niri_pass_count` | int | — | Total color-stage count |
| `niri_geo_size` | vec2 | — | Geometry size in logical pixels |
| `niri_corner_radius` | vec4 | — | Geometry corner radii |
| `niri_field` | sampler2D | TEXTURE1 | Final version-1 field |
| `niri_field_uv_rect` | vec4 | — | Visible crop in stable full-field UV; sample with `mix(niri_field_uv_rect.xy, niri_field_uv_rect.zw, v_coords)` |
| `niri_window_screen_rect` | vec4 | — | Window origin (xy) and size (zw) in screen UV |
| `niri_analytical_window` | int | — | `1` when the automatic source is the full-window analytical field, otherwise `0` |

## Rendering flow

```
Renderer/Layer render loop
  → background_effect::render_for_tile()
    → EffectBuffer or FramebufferEffect capture
    → Blur::render(options.shader_pipeline)

Blur::render_custom()
  → look up or compile the decoded version-1 color pipeline without filesystem access
  → reuse the cached automatic field when geometry inputs match
  → otherwise classify and generate one automatic field:
      - full window, including one client blur rectangle covering the full field → analytical rounded rectangle
      - independent simple rectangles → one instanced analytical draw
      - touching/overlapping/composite rectangles → bbox-local binary + JFA + Poisson
      - cursor silhouette → bbox-local cursor coverage + JFA + Poisson
  → run blur and shader color stages
  → for protocol regions and cursors, multiply final color by field A
  → return final texture; default Kawase remains the fallback on compile failure
```

`mask_texture_a` retains its historical name but holds the compositor-owned version-1 field.

The cursor shader pipeline reuses this machinery from `Niri::render_pointer`. The effect region is the cursor silhouette expanded by `cursor.effect-padding`, so `niri_color` includes backdrop beyond the silhouette while `niri_field` remains tight at `CursorEffectGeometry::coverage_bbox`. `FramebufferEffect::render` uses a per-output namespaced Id. The effect element replaces the plain cursor and is `Kind::Unspecified`, so it cannot reach a DRM cursor plane. Client `wl_surface` cursors and quarter-turn output transforms fall back to the plain cursor; Normal and Flipped180 are supported. Screenshots and screencasts pass `allow_cursor_effect = false` because their relocated pointer elements do not have the required backdrop.

## Caching

- **Custom programs:** `Shaders::custom_blur` checks a compact decoded-source/options key before resolving the compile configuration, then lazily caches `Option<CustomBlurProgram>`. Failures are cached as `None`; relevant global blur, window-rule, layer-rule, cursor-pipeline, and watched shader-file reloads clear the map.
- **Final field:** keyed by stable full-field size, normalized subregion `Arc` identity, cursor coverage identity, geometry size, and corner radii. Exact hits skip automatic generation independently of the selected color pipeline.
- **JFA output:** cached by bbox size, bbox-local rectangle offsets, and cursor identity. The bbox clamps both edges of the visible source before deriving its size, and the encoded bbox is copied one-to-one into the stable full-field texture; this prevents left/bottom clipping from rescaling the surviving field. Translation hits re-blit the encoded field without rerunning JFA/Poisson; when the same destination remains valid, only the previous bbox is cleared.
- **Cursor field:** cursor identity includes icon, scale, animation frame, and size. Hotspot-anchored motion deformation invalidates while the symmetric 55 ms velocity filter settles. Named shape transitions interpolate cached CPU R32F SDF endpoints in `mask_coverage.frag`.
- **Exact output clip:** protocol-region and cursor pipelines multiply final color by A from the stable full-field texture in `mask_output.frag`. The pass maps visible color UV through `niri_field_uv_rect`.
- **Normalized subregions:** `FramebufferEffect` caches normalized rectangles by source `Arc`, relative offset, scale, and full surface size. Sliding a partially offscreen effect changes only `niri_field_uv_rect`.
- **Programs and rectangle texture:** analytical, JFA, output-clip, and cursor-coverage programs compile once per `Blur`. The row-major RGBA32F rectangle texture grows but never shrinks and wraps across rows at `GL_MAX_TEXTURE_SIZE`.

## Error handling

- Missing or unsupported `version=` values, non-finite or non-positive shader scales, and pipelines without color stages fail pipeline resolution.
- Unreadable shader files fail config parsing while remaining watched, so creating or fixing the file triggers another reload.
- Shader compilation failures warn and fall through to the default Kawase blur.
- Compilation failures are cached as `None` to prevent retry storms.

## Testing

- **Visual tests:** `niri-visual-tests/` — GTK4/ADW app. `cargo run -p niri-visual-tests`
- **Shader pipeline examples:** `shaders/crt/`, `shaders/magnify/`, `shaders/overshifted3/`, `shaders/liquid-glass-faithful/`, `shaders/prism-glass/`, `shaders/cursor-glass/`, `shaders/debug-mask/`, `shaders/debug-edges/`, `shaders/jfa-debug/`. Color shaders use the same version-1 field texture across full windows, independent rectangles, composite regions, and cursors. `liquid-glass-faithful/glass.frag` uses `niri_analytical_window` for rounded analytical lighting. All shapes use bounded 256 px-gain, 64 px-cap physical displacement; the analytical window field itself supplies corner-following flow and reaches zero before its SDF Voronoi boundaries.
- **Config tests:** `insta` snapshot tests in `niri-config/`. Run with `cargo test -p niri-config`
- **Compile check:** `cargo check --all-targets`
- **Nested refresh:** the Winit backend reads the host monitor refresh when available. Set `NIRI_WINIT_REFRESH_RATE=<hz>` to override unreliable Wayland monitor metadata, for example `NIRI_WINIT_REFRESH_RATE=170 target/debug/niri --config config.kdl`.
- **Cava refresh:** `cava-test/shell.qml` follows the nested output refresh by default. Set `NIRI_CAVA_FRAMERATE=<fps>` to hold Cava at a separate rate when isolating compositor timing from region-update frequency.
- **Shader background:** `shader-bg-test/` renders the shadertoy shader in `truchet.frag` on a fullscreen background layer surface (namespace `shader-background`). `waves.frag` is kept as the previous wallpaper; point `shell.qml`'s `sourcePath` at it to switch back. `config.kdl` spawns it with `qs -n -p $HOME/projects/niri/shader-bg-test`, so a winit niri gets the background inside its own session. `compile-shader.sh` bakes the `.frag` into a `.qsb` (Qt 6 rejects raw GLSL) cached in `$XDG_STATE_HOME/quickshell/shaders`; it prefers `qsb` from PATH and otherwise takes the newest `qtshadertools` in the nix store, because a spawned shell inherits the session PATH rather than a quickshell wrapper's.
- **Donkey region test:** `donkey-test/DonkeyRegion.qml` contains a static 1900 px-wide protocol-region decomposition generated from the bundled `donkey.svg`; adjacent equal runs are coalesced. `shell.qml` uses the dedicated `donkey-test` namespace. `donkey-test/glass.frag` and `pipeline.kdl` are private copies of the liquid-glass pipeline with shader noise disabled, selected by a dedicated layer rule whose compositor post-process noise is also zero. No visual outline is drawn. Run it with `qs -p $HOME/projects/niri/donkey-test`.
- **Manual showcase:** `config.kdl` starts only `shader-bg-test`, and `Mod+Return` runs `showcase/terminal`, a reproducible transparent Kitty rooted at this checkout. Plain Kitty windows tile at one-third output width with pronounced rounded corners, drop shadows but no client decorations or compositor borders, and the global `liquid-glass-faithful` pipeline. From the last remaining terminal, run `showcase/1`, `showcase/2`, `showcase/3`, then `showcase/4`; they launch the aligned blank shape windows, `wl-bad-apple`, a top-right floating mpv Rickroll, and the Cava Quickshell layer respectively. The Cava launcher defaults `NIRI_CAVA_FRAMERATE` to 60. `NIRI_BAD_APPLE_PROJECT`, `NIRI_RICKROLL_SOURCE`, and an explicitly set `NIRI_CAVA_FRAMERATE` override their defaults.
- **Stained-glass lock UI:** `./lock.sh` launches the standalone native Rust client at `${NIRI_GLASS_LOCK_PROJECT:-$HOME/projects/niri-glass-lock}` through its Nix flake. Two top-level half-screen overlay layer surfaces move by layer-shell margins, each with one static SHM-rendered stained-glass buffer and a surface-local background-effect region. The `stained-glass-lock` rule uses the universal `prism-glass/window.frag` shader. It accepts any password and does not acquire a Wayland session lock.

## Precision notes

- Multigrid `u` textures use raw GL R32F. Half-float is insufficient near gradient minima.
- Multigrid RHS uses RG16F and bbox-local binary coverage uses R16F.
- RHS is scaled by `1/max_dist²` to keep `u` in a precision-friendly range.
- The multigrid pyramid grows dynamically until both coarse dimensions are at most 8 pixels, capped at 12 levels. The solver runs one V-cycle with three pre- and three post-Jacobi sweeps.
- The JFA encoder converts nearest-boundary distance to logical pixels, derives signed inward GB from the central-difference Poisson gradient, and fades confidence with both gradient strength and medial distance. Bbox-local binary coverage is copied to A; pixels below the 0.5 interior threshold retain fractional A with zero RGB. A cached 3×3 tent pass filters GB only, preserving distance and coverage.
- Bbox calculation has a 1e-4 epsilon snap to prevent pixel-coordinate wobble.
- `glColorMask(TRUE, FALSE, FALSE, FALSE)` is used during residual restrict to write only R.

## Update policy

When anything described in this file changes, update this AGENTS.md to reflect the new state.

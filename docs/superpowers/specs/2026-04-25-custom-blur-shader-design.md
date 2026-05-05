# Custom Blur Shader Pipeline

## Summary

Add a configurable custom blur shader system that allows users to replace the built-in Dual Kawase blur with their own multi-pass GLSL pipeline. Ship two example glassmorphic effects as demonstration.

## Motivation

Niri's current blur is a Dual Kawase algorithm (downsample/upsample) that produces a uniform Gaussian-like blur. This is sufficient for basic window backdrop effects but cannot produce more sophisticated visual effects like:

- Physically-based glass simulation with refraction
- Apple-like "Liquid Glass" with edge highlights and specular effects
- Chromatic aberration, caustics, or other optical phenomena

These effects require warping the background before or during the blur (refraction), varying blur intensity spatially (edge-aware), and applying per-channel color effects (chromatic aberration) — none of which the Kawase pipeline supports.

The existing custom animation shader system (for resize/close/open) demonstrates that niri can safely load user-provided GLSL. This design extends that pattern to the blur pipeline, but with full multi-pass control since glassmorphic effects fundamentally require multiple rendering passes with different scale factors.

## Config Schema

A new `custom-shader` field on the existing `Blur` config struct:

```kdl
blur {
    passes 3
    offset 3.
    noise 0.02
    saturation 1.5
    custom-shader "/path/to/my-glass-shader"
}
```

When `custom-shader` is set, the built-in Kawase pipeline is completely bypassed. The `passes` and `offset` fields are ignored (they are Kawase-specific). The `noise` and `saturation` fields still apply — they are handled by the post-processing/compositing step that runs after the blur/effect pipeline.

The path can be absolute or relative to the config file directory, matching the existing `custom-shader` convention for animation shaders.

Reloaded on config reload (same mechanism as animation custom shaders). Compilation errors produce a warning log and fall back to the built-in Kawase blur.

## Pipeline Manifest Format

The directory pointed to by `custom-shader` contains a `pipeline.kdl` manifest and individual `.frag` files:

```
my-glass-shader/
  pipeline.kdl
  refract.frag
  blur_down.frag
  blur_up.frag
  composite.frag
```

Manifest format:

```kdl
pass "refract" file="refract.frag" scale=1.0
pass "blur_down" file="blur_down.frag" scale=0.5
pass "blur_up" file="blur_up.frag" scale=2.0
pass "composite" file="composite.frag" scale=1.0
```

Each `pass` node:
- **`name`** (positional string): Human-readable pass name, used in log messages.
- **`file`** (property): Path to the `.frag` file, relative to the manifest directory.
- **`scale`** (property float): Output texture size relative to the input texture size. For the first pass, input is the full-resolution background capture. Subsequent passes receive the previous pass's output.

Scale examples:
- `scale=1.0`: Output same size as input (full-resolution processing)
- `scale=0.5`: Output half the input size (downsample)
- `scale=2.0`: Output double the input size (upsample)

Intermediate texture sizes are computed by chaining scale factors. The final pass's output is the result that gets composited.

## Per-pass Shader Interface

Each `.frag` file is a standalone GLSL ES 1.00 fragment shader with its own `main()`. Niri provides these uniforms to every pass:

```glsl
// The input texture (previous pass output, or background for pass 0).
uniform sampler2D niri_input;

// Size of the output texture in pixels.
uniform vec2 niri_output_size;

// Size of the input texture in pixels.
uniform vec2 niri_input_size;

// Half pixel in output space: vec2(0.5) / niri_output_size.
uniform vec2 niri_half_pixel;

// Current pass index (0-based).
uniform int niri_pass;

// Total number of passes.
uniform int niri_pass_count;

// Original blur region size in pixels (before any pass scaling).
// Use this (not niri_output_size) for edge-distance calculations.
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;

// Standard [0,1] texture coordinates from vertex shader.
varying vec2 v_coords;
```

The vertex shader is the existing `blur.vert` (maps `v_coords` from `[0,1]`).

User writes a normal fragment shader:

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_half_pixel;
uniform vec2 niri_output_size;

void main() {
    vec4 color = texture2D(niri_input, v_coords);
    gl_FragColor = color;
}
```

Each pass is compiled into a separate GL program at load time (`.frag` + shared `blur.vert`).

## Rust-side Architecture

### New types (in `blur.rs`)

```rust
struct CustomBlurPassConfig {
    name: String,
    file: PathBuf,
    scale: f32,
}

struct CustomBlurPassProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_input_size: ffi::types::GLint,
    uniform_half_pixel: ffi::types::GLint,
    uniform_pass: ffi::types::GLint,
    uniform_pass_count: ffi::types::GLint,
    uniform_geo_size: ffi::types::GLint,
    uniform_corner_radius: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

struct CustomBlurProgram {
    passes: Vec<CustomBlurPassProgram>,
    config: Vec<CustomBlurPassConfig>,
}
```

### Changes to `Shaders` struct

Add a `custom_blur: RefCell<Option<CustomBlurProgram>>` field, following the same `RefCell<Option<...>>` pattern as `custom_resize`/`custom_close`/`custom_open`.

Add `replace_custom_blur_program()` method, same pattern as the animation shader replace methods.

### Execution flow

`Blur::render()` remains the single entry point. It branches internally:

1. **Custom path** (when `custom_blur` is set):
   - Compute intermediate texture sizes by chaining `scale` factors from the manifest
   - Create/reuse FBOs at each computed size (same lazy allocation as today)
   - For each pass: bind input texture, set output FBO, set all uniforms, draw fullscreen quad
   - Return the final pass's output texture

2. **Built-in path** (when no custom program): existing Kawase pipeline, unchanged.

This means `FramebufferEffectElement` and `EffectBuffer` (xray path) need no changes — they already call `Blur::render()`.

### Config parsing

New `custom_shader: Option<String>` field on the `Blur`/`BlurPart` config structs in `niri-config/src/appearance.rs`. Parsed as a `knuffel` child node `custom-shader` with a string argument.

### Config reload

In `niri.rs`, same pattern as animation shaders (lines 1548-1573): when `blur.custom_shader` changes, load the manifest from disk, compile all passes, and swap the program via `Shaders::replace_custom_blur_program()`. Compilation errors warn and fall back to Kawase.

### Manifest parsing

A new function reads `pipeline.kdl` from the shader directory, parses each `pass` node, and returns `Vec<CustomBlurPassConfig>`. Uses the same `knuffel` parser as the rest of niri's config.

## Example Shaders

Two example shader sets in `docs/wiki/examples/`:

### Example 1: Physically-based glass simulation (`glass-physics/`)

3-pass pipeline:

```
glass-physics/
  pipeline.kdl
  refract.frag      // Distort background via refraction map
  blur.frag         // Single-pass gather blur (disc sample pattern)
  composite.frag    // Chromatic aberration, specular highlights
```

- **Pass 0 (refract)**: Full resolution, scale=1.0. Computes refraction offset based on distance from blur region edges (thicker glass at borders). Warps texture lookups.
- **Pass 1 (blur)**: Full resolution, scale=1.0. Circular disc sample pattern from the refracted texture. Sample count configurable in the shader.
- **Pass 2 (composite)**: Full resolution, scale=1.0. Per-channel color offset for chromatic aberration (stronger near edges), specular highlight from simulated surface normals, saturation adjustment.

### Example 2: Liquid Glass / Apple-like (`glass-liquid/`)

4-pass pipeline:

```
glass-liquid/
  pipeline.kdl
  refract.frag      // Lens-like distortion
  blur_down.frag    // Downsample blur pass
  blur_up.frag      // Upsample blur pass
  composite.frag    // Edge highlights, specular bloom
```

- **Pass 0 (refract)**: Full resolution, scale=1.0. Lens-barrel distortion, stronger at edges, subtle at center.
- **Pass 1 (blur_down)**: Half resolution, scale=0.5. Modified Kawase downsample with wider sampling.
- **Pass 2 (blur_up)**: Full resolution, scale=2.0. Upsample with wider pattern.
- **Pass 3 (composite)**: Full resolution, scale=1.0. Bright edge highlights (light catching glass rim), specular bloom, chromatic fringing at borders, subtle environment reflection tint.

Both examples will have inline comments documenting each uniform and how to tune parameters.

## Files Changed

### Modified files
- `niri-config/src/appearance.rs` — Add `custom_shader` field to `Blur`/`BlurPart`, add parsing
- `src/render_helpers/blur.rs` — Add `CustomBlurProgram`, `CustomBlurPassProgram`, custom rendering path in `Blur::render()`
- `src/render_helpers/shaders/mod.rs` — Add `custom_blur` field to `Shaders`, add `replace_custom_blur_program()` and `set_custom_blur_program()`
- `src/niri.rs` — Add config reload handling for `blur.custom_shader`
- `src/backend/tty.rs` — Initial load of custom blur shader on startup
- `src/backend/winit.rs` — Initial load of custom blur shader on startup

### New files
- `docs/wiki/examples/glass-physics/pipeline.kdl`
- `docs/wiki/examples/glass-physics/refract.frag`
- `docs/wiki/examples/glass-physics/blur.frag`
- `docs/wiki/examples/glass-physics/composite.frag`
- `docs/wiki/examples/glass-liquid/pipeline.kdl`
- `docs/wiki/examples/glass-liquid/refract.frag`
- `docs/wiki/examples/glass-liquid/blur_down.frag`
- `docs/wiki/examples/glass-liquid/blur_up.frag`
- `docs/wiki/examples/glass-liquid/composite.frag`

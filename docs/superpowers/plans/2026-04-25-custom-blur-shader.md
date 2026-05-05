# Custom Blur Shader Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable custom blur shader system that replaces the built-in Dual Kawase blur with user-defined multi-pass GLSL pipelines, plus two example glassmorphic effects.

**Architecture:** The custom pipeline is stored as a directory containing a `pipeline.kdl` manifest and per-pass `.frag` files. The existing `Blur::render()` method branches to a custom execution path when a custom program is loaded. The config field `custom-shader` under `blur {}` stores the directory path (not inline GLSL, unlike animation shaders, because we need multiple files).

**Tech Stack:** Rust, GLSL ES 1.00, OpenGL ES 2.0 (via smithay's GlesRenderer), knuffel (KDL parsing).

**Spec:** `docs/superpowers/specs/2026-04-25-custom-blur-shader-design.md`

---

### Task 1: Config — Add `custom_shader` field to Blur structs

**Files:**
- Modify: `niri-config/src/appearance.rs:1009-1056`

- [ ] **Step 1: Add `custom_shader` field to the `Blur` and `BlurPart` structs**

In `niri-config/src/appearance.rs`, add `custom_shader: Option<String>` to the `Blur` struct (line ~1015, after `saturation`):

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blur {
    pub off: bool,
    pub passes: u8,
    pub offset: f64,
    pub noise: f64,
    pub saturation: f64,
    pub custom_shader: Option<String>,
}
```

Update `Default` impl (line ~1018):

```rust
impl Default for Blur {
    fn default() -> Self {
        Self {
            off: false,
            passes: 3,
            offset: 3.,
            noise: 0.02,
            saturation: 1.5,
            custom_shader: None,
        }
    }
}
```

Add to `BlurPart` (line ~1031):

```rust
#[derive(knuffel::Decode, Debug, Default, Clone, Copy, PartialEq)]
pub struct BlurPart {
    #[knuffel(child)]
    pub off: bool,
    #[knuffel(child)]
    pub on: bool,
    #[knuffel(child, unwrap(argument))]
    pub passes: Option<u8>,
    #[knuffel(child, unwrap(argument))]
    pub offset: Option<FloatOrInt<0, 100>>,
    #[knuffel(child, unwrap(argument))]
    pub noise: Option<FloatOrInt<0, 1000>>,
    #[knuffel(child, unwrap(argument))]
    pub saturation: Option<FloatOrInt<0, 1000>>,
    #[knuffel(child, unwrap(argument))]
    pub custom_shader: Option<String>,
}
```

Update the `MergeWith<BlurPart>` impl (line ~1046) to merge `custom_shader`:

```rust
impl MergeWith<BlurPart> for Blur {
    fn merge_with(&mut self, part: &BlurPart) {
        self.off |= part.off;
        if part.on {
            self.off = false;
        }

        merge_clone!((self, part), passes);
        merge!((self, part), offset, noise, saturation);
        merge_clone_opt!((self, part), custom_shader);
    }
}
```

Note: `custom_shader` uses `Option<String>` which is `Copy`-incompatible. Since `Blur` currently derives `Copy`, we need to remove `Copy` from `Blur`. Check all usages of `Blur` that rely on `Copy` and update them. The `merge_clone_opt!` macro should work with `Clone`. If `Blur` is used with `.copy()` anywhere, change those to `.clone()`.

- [ ] **Step 2: Run `cargo check -p niri-config` to verify compilation**

Run: `cargo check -p niri-config`
Expected: Compiles successfully (may have warnings about unused field, that's fine).

- [ ] **Step 3: Commit**

```bash
git add niri-config/src/appearance.rs
git commit -m "feat(config): add custom-shader field to blur config"
```

---

### Task 2: Pipeline manifest parser

**Files:**
- Create: `src/render_helpers/custom_blur.rs`

This module parses the `pipeline.kdl` manifest from a shader directory and loads all `.frag` file contents.

- [ ] **Step 1: Create `src/render_helpers/custom_blur.rs` with manifest types and parser**

```rust
use std::path::{Path, PathBuf};

use anyhow::{Context as _, ensure};
use knuffel::Decode as _;

#[derive(Debug, Clone)]
pub struct CustomBlurPassConfig {
    pub name: String,
    pub source: String,
    pub scale: f32,
}

#[derive(knuffel::Decode, Debug)]
struct PipelinePass {
    #[knuffel(argument)]
    name: String,
    #[knuffel(property)]
    file: String,
    #[knuffel(property, default = 1.0)]
    scale: f32,
}

#[derive(knuffel::Decode, Debug)]
struct PipelineManifest {
    #[knuffel(children)]
    passes: Vec<PipelinePass>,
}

pub fn load_custom_blur_pipeline(
    dir: &Path,
) -> anyhow::Result<Vec<CustomBlurPassConfig>> {
    ensure!(dir.is_dir(), "custom shader path is not a directory: {}", dir.display());

    let manifest_path = dir.join("pipeline.kdl");
    let manifest_text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;

    let manifest: PipelineManifest = knuffel::parse(&manifest_path.to_string_lossy(), &manifest_text)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;

    ensure!(!manifest.passes.is_empty(), "pipeline must have at least one pass");

    let mut passes = Vec::with_capacity(manifest.passes.len());
    for (i, pass) in manifest.passes.iter().enumerate() {
        ensure!(pass.scale > 0.0, "pass {} ({:?}): scale must be positive", i, pass.name);

        let frag_path = dir.join(&pass.file);
        let source = std::fs::read_to_string(&frag_path)
            .with_context(|| format!("failed to read pass {} ({:?}): {}", i, pass.name, frag_path.display()))?;

        passes.push(CustomBlurPassConfig {
            name: pass.name.clone(),
            source,
            scale: pass.scale,
        });
    }

    Ok(passes)
}
```

- [ ] **Step 2: Add `mod custom_blur` to `src/render_helpers/mod.rs`**

Add `pub mod custom_blur;` to the render_helpers module file.

- [ ] **Step 3: Run `cargo check -p niri` to verify compilation**

Run: `cargo check -p niri`
Expected: Compiles successfully.

- [ ] **Step 4: Commit**

```bash
git add src/render_helpers/custom_blur.rs src/render_helpers/mod.rs
git commit -m "feat: add pipeline manifest parser for custom blur shaders"
```

---

### Task 3: CustomBlurProgram — GL compilation and types

**Files:**
- Modify: `src/render_helpers/blur.rs`

Add the GL program types and compilation logic for custom blur passes. Uses `Rc` wrapping (same pattern as `BlurProgram`) so the program can be cheaply cloned from the `RefCell` without borrow conflicts.

- [ ] **Step 1: Add `CustomBlurPassProgram` and `CustomBlurProgram` types to `blur.rs`**

Add after the existing `BlurProgram` impl block (after line 94):

```rust
#[derive(Debug)]
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

#[derive(Debug)]
struct CustomBlurProgramInner {
    passes: Vec<CustomBlurPassProgram>,
    scales: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CustomBlurProgram(Rc<CustomBlurProgramInner>);
```

- [ ] **Step 2: Add compilation function**

```rust
unsafe fn compile_custom_pass(
    gl: &ffi::Gles2,
    frag_src: &str,
) -> Result<CustomBlurPassProgram, GlesError> {
    let vert_src = include_str!("shaders/blur.vert");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };

    let input = c"niri_input";
    let output_size = c"niri_output_size";
    let input_size = c"niri_input_size";
    let half_pixel = c"niri_half_pixel";
    let pass = c"niri_pass";
    let pass_count = c"niri_pass_count";
    let geo_size = c"niri_geo_size";
    let corner_radius = c"niri_corner_radius";
    let vert = c"vert";

    Ok(CustomBlurPassProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, input.as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, output_size.as_ptr()),
        uniform_input_size: gl.GetUniformLocation(program, input_size.as_ptr()),
        uniform_half_pixel: gl.GetUniformLocation(program, half_pixel.as_ptr()),
        uniform_pass: gl.GetUniformLocation(program, pass.as_ptr()),
        uniform_pass_count: gl.GetUniformLocation(program, pass_count.as_ptr()),
        uniform_geo_size: gl.GetUniformLocation(program, geo_size.as_ptr()),
        uniform_corner_radius: gl.GetUniformLocation(program, corner_radius.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}
```

- [ ] **Step 3: Add `CustomBlurProgram::compile` and `destroy`**

```rust
impl CustomBlurProgram {
    pub fn compile(
        renderer: &mut GlesRenderer,
        pass_configs: &[custom_blur::CustomBlurPassConfig],
    ) -> anyhow::Result<Self> {
        let scales: Vec<f32> = pass_configs.iter().map(|c| c.scale).collect();
        renderer
            .with_context(move |gl| unsafe {
                let mut passes = Vec::with_capacity(pass_configs.len());
                for (i, config) in pass_configs.iter().enumerate() {
                    let pass = compile_custom_pass(gl, &config.source)
                        .with_context(|| format!("error compiling custom blur pass {} ({:?})", i, config.name))?;
                    passes.push(pass);
                }
                Ok(Self(Rc::new(CustomBlurProgramInner { passes, scales })))
            })
            .context("error making GL context current")?
    }

    pub fn destroy(self, renderer: &mut GlesRenderer) -> Result<(), GlesError> {
        renderer.with_context(move |gl| unsafe {
            for pass in &self.0.passes {
                gl.DeleteProgram(pass.program);
            }
        })
    }
}
```

- [ ] **Step 4: Run `cargo check -p niri`**

Run: `cargo check -p niri`
Expected: Compiles successfully.

- [ ] **Step 5: Commit**

```bash
git add src/render_helpers/blur.rs
git commit -m "feat: add CustomBlurProgram types and GL compilation"
```

---

### Task 4: Shaders struct — integrate custom blur program

**Files:**
- Modify: `src/render_helpers/shaders/mod.rs`

Add the `custom_blur` field to `Shaders`, plus the set/replace functions.

- [ ] **Step 1: Add `custom_blur` field to `Shaders` and `replace_custom_blur_program` method**

In `src/render_helpers/shaders/mod.rs`, add the field to `Shaders` (after `custom_open`):

```rust
pub struct Shaders {
    pub border: Option<ShaderProgram>,
    pub shadow: Option<ShaderProgram>,
    pub clipped_surface: Option<GlesTexProgram>,
    pub postprocess_and_clip: Option<GlesTexProgram>,
    pub resize: Option<ShaderProgram>,
    pub gradient_fade: Option<GlesTexProgram>,
    pub blur: Option<BlurProgram>,
    pub custom_resize: RefCell<Option<ShaderProgram>>,
    pub custom_close: RefCell<Option<ShaderProgram>>,
    pub custom_open: RefCell<Option<ShaderProgram>>,
    pub custom_blur: RefCell<Option<super::blur::CustomBlurProgram>>,
}
```

Add the import at the top if needed (the `use super::blur::BlurProgram` is already there, so `super::blur::CustomBlurProgram` should work).

Update `Shaders::compile()` to initialize the new field:

```rust
custom_blur: RefCell::new(None),
```

Add `replace_custom_blur_program` method:

```rust
pub fn replace_custom_blur_program(
    &self,
    program: Option<super::blur::CustomBlurProgram>,
) -> Option<super::blur::CustomBlurProgram> {
    self.custom_blur.replace(program)
}
```

- [ ] **Step 2: Add `set_custom_blur_program` function**

Add after the existing `set_custom_open_program` function:

```rust
pub fn set_custom_blur_program(
    renderer: &mut GlesRenderer,
    dir: Option<&str>,
) {
    let program = if let Some(dir) = dir {
        let path = std::path::Path::new(dir);
        match super::custom_blur::load_custom_blur_pipeline(path) {
            Ok(configs) => {
                if configs.is_empty() {
                    warn!("custom blur pipeline has no passes, ignoring");
                    None
                } else {
                    match super::blur::CustomBlurProgram::compile(renderer, &configs) {
                        Ok(program) => {
                            info!("loaded custom blur shader with {} passes from {}", configs.len(), dir);
                            Some(program)
                        }
                        Err(err) => {
                            warn!("error compiling custom blur shader: {err:?}");
                            None
                        }
                    }
                }
            }
            Err(err) => {
                warn!("error loading custom blur shader from {}: {err:?}", dir);
                None
            }
        }
    } else {
        None
    };

    if let Some(prev) = Shaders::get(renderer).replace_custom_blur_program(program) {
        if let Err(err) = prev.destroy(renderer) {
            warn!("error destroying previous custom blur shader: {err:?}");
        }
    }
}
```

- [ ] **Step 3: Run `cargo check -p niri`**

Run: `cargo check -p niri`
Expected: Compiles successfully.

- [ ] **Step 4: Commit**

```bash
git add src/render_helpers/shaders/mod.rs
git commit -m "feat: integrate custom blur program into Shaders struct"
```

---

### Task 5: Blur::render() — custom pipeline execution

**Files:**
- Modify: `src/render_helpers/blur.rs`
- Modify: `src/render_helpers/framebuffer_effect.rs`
- Modify: `src/render_helpers/effect_buffer.rs` (if it uses Blur)

Add the custom pipeline execution path to `Blur::render()`. When a custom program exists in `Shaders`, use it instead of the Kawase pipeline. The custom program is cloned (Rc only) from the `RefCell` to avoid borrow conflicts — same pattern used by `BlurProgram`.

- [ ] **Step 1: Add `custom_textures` field to `Blur` struct**

The custom pipeline needs separate texture storage since sizes differ from Kawase:

```rust
#[derive(Debug)]
pub struct Blur {
    program: BlurProgram,
    renderer_context_id: ContextId<GlesTexture>,
    textures: Vec<GlesTexture>,
    custom_textures: Vec<GlesTexture>,
}
```

Update `Blur::new()`:

```rust
pub fn new(renderer: &mut GlesRenderer) -> Option<Self> {
    let program = Shaders::get(renderer).blur.clone()?;
    Some(Self {
        program,
        renderer_context_id: renderer.context_id(),
        textures: Vec::new(),
        custom_textures: Vec::new(),
    })
}
```

- [ ] **Step 2: Update `BlurOptions` with geometry fields**

Custom blur shaders need the blur region geometry for edge-aware effects. Add fields to `BlurOptions`:

```rust
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct BlurOptions {
    pub passes: u8,
    pub offset: f64,
    pub geo_size: (f32, f32),
    pub corner_radius: [f32; 4],
}

impl From<niri_config::Blur> for BlurOptions {
    fn from(config: niri_config::Blur) -> Self {
        Self {
            passes: config.passes,
            offset: config.offset,
            geo_size: (0.0, 0.0),
            corner_radius: [0.0; 4],
        }
    }
}

impl BlurOptions {
    pub fn with_geometry(mut self, geo_size: (f32, f32), corner_radius: [f32; 4]) -> Self {
        self.geo_size = geo_size;
        self.corner_radius = corner_radius;
        self
    }
}
```

- [ ] **Step 3: Add `render_custom` method to `Blur`**

This method executes the multi-pass custom pipeline. It manages intermediate textures and chains passes together:

```rust
fn render_custom(
    &mut self,
    renderer: &mut GlesRenderer,
    source: &GlesTexture,
    custom_program: &CustomBlurProgram,
    geo_size: (f32, f32),
    corner_radius: [f32; 4],
) -> anyhow::Result<GlesTexture> {
    let _span = tracy_client::span!("Blur::render_custom");
    trace!("rendering custom blur");

    ensure!(
        renderer.context_id() == self.renderer_context_id,
        "wrong renderer"
    );

    let passes = &custom_program.0.passes;
    let scales = &custom_program.0.scales;
    let pass_count = passes.len();

    let source_size = source.size();
    let source_w = source_size.w as f32;
    let source_h = source_size.h as f32;

    let mut pass_output_sizes: Vec<(i32, i32)> = Vec::with_capacity(pass_count);
    let mut current_w = source_w;
    let mut current_h = source_h;
    for &scale in scales {
        current_w = (current_w * scale).max(1.0);
        current_h = (current_h * scale).max(1.0);
        pass_output_sizes.push((current_w as i32, current_h as i32));
    }

    let needs_recreate = self.custom_textures.len() != pass_count
        || self.custom_textures.iter().zip(pass_output_sizes.iter())
            .any(|(tex, &size)| tex.size().w != size.0 || tex.size().h != size.1);

    if needs_recreate {
        self.custom_textures.clear();
        for &(w, h) in &pass_output_sizes {
            let size = Size::new(w, h);
            let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, size)?;
            self.custom_textures.push(texture);
        }
    }

    renderer.with_profiled_context(gpu_span_location!("Blur::render_custom"), |gl| unsafe {
        while gl.GetError() != ffi::NO_ERROR {}

        gl.Disable(ffi::BLEND);
        gl.Disable(ffi::SCISSOR_TEST);
        gl.ActiveTexture(ffi::TEXTURE0);

        let mut fbos = [0; 2];
        gl.GenFramebuffers(fbos.len() as _, fbos.as_mut_ptr());
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbos[0]);

        let vertices: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

        for (i, pass) in passes.iter().enumerate() {
            let (output_w, output_h) = pass_output_sizes[i];
            let (input_w, input_h) = if i == 0 {
                (source_size.w, source_size.h)
            } else {
                let prev = pass_output_sizes[i - 1];
                (prev.0, prev.1)
            };

            gl.UseProgram(pass.program);
            gl.Uniform1i(pass.uniform_input, 0);
            gl.Uniform2f(pass.uniform_output_size, output_w as f32, output_h as f32);
            gl.Uniform2f(pass.uniform_input_size, input_w as f32, input_h as f32);
            gl.Uniform2f(pass.uniform_half_pixel, 0.5 / output_w as f32, 0.5 / output_h as f32);
            gl.Uniform1i(pass.uniform_pass, i as i32);
            gl.Uniform1i(pass.uniform_pass_count, pass_count as i32);
            gl.Uniform2f(pass.uniform_geo_size, geo_size.0, geo_size.1);
            gl.Uniform4f(pass.uniform_corner_radius, corner_radius[0], corner_radius[1], corner_radius[2], corner_radius[3]);

            gl.Viewport(0, 0, output_w, output_h);

            gl.EnableVertexAttribArray(pass.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                pass.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                vertices.as_ptr().cast(),
            );

            let dst = self.custom_textures[i].tex_id();
            let src = if i == 0 {
                source.tex_id()
            } else {
                self.custom_textures[i - 1].tex_id()
            };

            trace!("custom blur pass {i}: drawing {src} to {dst}");

            gl.FramebufferTexture2D(
                ffi::DRAW_FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                dst,
                0,
            );

            gl.BindTexture(ffi::TEXTURE_2D, src);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);

            gl.DrawArrays(ffi::TRIANGLES, 0, 6);
            gl.DisableVertexAttribArray(pass.attrib_vert as u32);
        }

        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);
        gl.DeleteFramebuffers(fbos.len() as _, fbos.as_ptr());
    })?;

    Ok(self.custom_textures.last().unwrap().clone())
}
```

- [ ] **Step 4: Modify `Blur::render()` to branch on custom program**

At the start of the existing `Blur::render()` method (line 166), add the custom program check before the Kawase code. The `CustomBlurProgram` uses `Rc` so cloning is cheap (just a reference count bump):

```rust
pub fn render(
    &mut self,
    renderer: &mut GlesRenderer,
    source: &GlesTexture,
    options: BlurOptions,
) -> anyhow::Result<GlesTexture> {
    // Check for custom blur program (Rc clone avoids borrow conflict).
    let custom = Shaders::get(renderer).custom_blur.borrow().clone();
    if let Some(custom) = custom {
        return self.render_custom(renderer, source, &custom, options.geo_size, options.corner_radius);
    }

    // Existing Kawase code continues unchanged below...
```

- [ ] **Step 5: Update callers to pass geometry in `BlurOptions`**

In `src/render_helpers/framebuffer_effect.rs`, the `capture_framebuffer()` method has `self.geometry` and `self.corner_radius`. Where `blur_options` is created and used (around line 71 and 236-314), enrich it with geometry:

```rust
// In the render() method (line 71), where blur_options is set:
blur_options: self.blur_options.map(|opts| {
    let geo = params.geometry.size;
    opts.with_geometry(
        (geo.w as f32, geo.h as f32),
        <[f32; 4]>::from(params.clip.map(|(geo, cr)| cr).unwrap_or_default()),
    )
}),
```

Note: the exact geometry extraction depends on how `params.clip` provides corner radius. Check the existing `render()` method in `background_effect.rs` for how `clip_geo` and `corner_radius` are extracted from `params`, and use the same values.

Similarly, check `src/render_helpers/effect_buffer.rs` and `src/render_helpers/xray.rs` for any `Blur::render()` calls and add geometry to `BlurOptions`.

- [ ] **Step 6: Run `cargo check -p niri`**

Run: `cargo check -p niri`
Expected: Compiles successfully.

- [ ] **Step 7: Commit**

```bash
git add src/render_helpers/blur.rs src/render_helpers/framebuffer_effect.rs
git commit -m "feat: add custom pipeline execution path to Blur::render()"
```

---

### Task 6: Startup and config reload hooks

**Files:**
- Modify: `src/niri.rs`
- Modify: `src/backend/tty.rs`
- Modify: `src/backend/winit.rs`

- [ ] **Step 1: Add initial load in `tty.rs`**

In `src/backend/tty.rs`, after the existing custom animation shader loads (around line 845), add:

```rust
if let Some(dir) = config.blur.custom_shader.as_deref() {
    shaders::set_custom_blur_program(gles_renderer, Some(dir));
}
```

- [ ] **Step 2: Add initial load in `winit.rs`**

In `src/backend/winit.rs`, after the existing custom animation shader loads (around line 169), add:

```rust
if let Some(dir) = config.animations.window_open.custom_shader.as_deref() {
    shaders::set_custom_open_program(renderer, Some(dir));
}
if let Some(dir) = config.blur.custom_shader.as_deref() {
    shaders::set_custom_blur_program(renderer, Some(dir));
}
```

Wait, the first line already exists. Just add the blur line.

- [ ] **Step 3: Add config reload in `niri.rs`**

In `src/niri.rs`, after the existing custom shader reload blocks (around line 1576), add:

```rust
if config.blur.custom_shader != old_config.blur.custom_shader {
    let dir = config.blur.custom_shader.as_deref();
    self.backend.with_primary_renderer(|renderer| {
        shaders::set_custom_blur_program(renderer, dir);
    });
    shaders_changed = true;
}
```

- [ ] **Step 4: Run `cargo check -p niri`**

Run: `cargo check -p niri`
Expected: Compiles successfully.

- [ ] **Step 5: Commit**

```bash
git add src/niri.rs src/backend/tty.rs src/backend/winit.rs
git commit -m "feat: add startup and reload hooks for custom blur shader"
```

---

### Task 7: Build and smoke test

**Files:**
- No new files

- [ ] **Step 1: Build the project**

Run: `cargo build`
Expected: Builds successfully.

- [ ] **Step 2: Run existing tests**

Run: `cargo test`
Expected: All existing tests pass. No new tests break.

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address compilation/test issues from custom blur integration"
```

---

### Task 8: Example shader — Physically-based glass simulation

**Files:**
- Create: `docs/wiki/examples/glass-physics/pipeline.kdl`
- Create: `docs/wiki/examples/glass-physics/refract.frag`
- Create: `docs/wiki/examples/glass-physics/blur.frag`
- Create: `docs/wiki/examples/glass-physics/composite.frag`

- [ ] **Step 1: Create `pipeline.kdl`**

```kdl
pass "refract" file="refract.frag" scale=1.0
pass "blur" file="blur.frag" scale=1.0
pass "composite" file="composite.frag" scale=1.0
```

- [ ] **Step 2: Create `refract.frag`**

This pass warps the background texture based on distance from the edges of the blur region, simulating light refracting through glass that is thicker at the edges.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    // Distance from center, normalized to [0, 0.5] range.
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0; // 0 at center, ~1.41 at corners

    // Refraction strength increases toward edges.
    // This simulates a glass pane that is thicker at the borders.
    float refraction_strength = smoothstep(0.0, 1.0, dist) * 0.02;

    // Compute refraction offset: push coordinates away from center.
    vec2 offset = center * refraction_strength;

    vec2 refracted_coords = v_coords + offset;

    gl_FragColor = texture2D(niri_input, refracted_coords);
}
```

- [ ] **Step 3: Create `blur.frag`**

Single-pass disc-pattern gather blur for a smooth glass-like effect.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

// Golden spiral disc sampling for a smooth circular blur.
// Adjust SAMPLE_COUNT for quality vs performance.
#define SAMPLE_COUNT 16

void main() {
    float radius = 4.0; // Blur radius in pixels
    vec4 color = vec4(0.0);
    float total_weight = 0.0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        // Golden angle spiral distribution.
        float angle = float(i) * 2.39996323;
        float r = radius * sqrt(float(i) / float(SAMPLE_COUNT)) / niri_output_size.x;
        vec2 offset = vec2(cos(angle), sin(angle)) * r;

        // Gaussian-ish weight: stronger at center.
        float weight = 1.0 - float(i) / float(SAMPLE_COUNT);
        color += texture2D(niri_input, v_coords + offset) * weight;
        total_weight += weight;
    }

    gl_FragColor = color / total_weight;
}
```

- [ ] **Step 4: Create `composite.frag`**

Applies chromatic aberration (stronger near edges), specular highlights, and final compositing.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0;

    // Chromatic aberration: offset R and B channels in opposite directions.
    // Stronger near edges.
    float chromatic_strength = smoothstep(0.3, 1.0, dist) * 0.003;
    vec2 chromatic_offset = normalize(center + 0.001) * chromatic_strength;

    float r = texture2D(niri_input, v_coords + chromatic_offset).r;
    float g = texture2D(niri_input, v_coords).g;
    float b = texture2D(niri_input, v_coords - chromatic_offset).b;
    float a = texture2D(niri_input, v_coords).a;

    vec4 color = vec4(r, g, b, a);

    // Specular highlight: simulate light from the top-left.
    float specular = pow(max(0.0, 1.0 - dist), 8.0) * 0.15;
    color.rgb += vec3(specular);

    // Slight brightness boost at edges to simulate glass rim.
    float rim = smoothstep(0.6, 1.0, dist) * 0.05;
    color.rgb += vec3(rim);

    gl_FragColor = color;
}
```

- [ ] **Step 5: Commit**

```bash
git add docs/wiki/examples/glass-physics/
git commit -m "feat: add physically-based glass simulation example shader"
```

---

### Task 9: Example shader — Liquid Glass (Apple-like)

**Files:**
- Create: `docs/wiki/examples/glass-liquid/pipeline.kdl`
- Create: `docs/wiki/examples/glass-liquid/refract.frag`
- Create: `docs/wiki/examples/glass-liquid/blur_down.frag`
- Create: `docs/wiki/examples/glass-liquid/blur_up.frag`
- Create: `docs/wiki/examples/glass-liquid/composite.frag`

- [ ] **Step 1: Create `pipeline.kdl`**

```kdl
pass "refract" file="refract.frag" scale=1.0
pass "blur_down" file="blur_down.frag" scale=0.5
pass "blur_up" file="blur_up.frag" scale=2.0
pass "composite" file="composite.frag" scale=1.0
```

- [ ] **Step 2: Create `refract.frag`**

Stronger lens-barrel distortion at edges, subtle at center.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    vec2 center = v_coords - 0.5;
    float r2 = dot(center, center); // r squared
    float r4 = r2 * r2;

    // Barrel distortion: stronger at edges.
    float k1 = -0.1;  // Primary distortion coefficient
    float k2 = 0.05;  // Secondary distortion coefficient
    float distortion = 1.0 + k1 * r2 + k2 * r4;

    vec2 distorted = center * distortion + 0.5;

    gl_FragColor = texture2D(niri_input, distorted);
}
```

- [ ] **Step 3: Create `blur_down.frag`**

Modified Kawase downsample with wider sampling for smoother base.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_half_pixel;

void main() {
    vec2 o = niri_half_pixel * 4.0; // Wider sampling than standard Kawase
    vec4 sum = texture2D(niri_input, v_coords) * 4.0;
    sum += texture2D(niri_input, v_coords + vec2(-o.x, -o.y));
    sum += texture2D(niri_input, v_coords + vec2( o.x, -o.y));
    sum += texture2D(niri_input, v_coords + vec2(-o.x,  o.y));
    sum += texture2D(niri_input, v_coords + vec2( o.x,  o.y));
    gl_FragColor = sum / 8.0;
}
```

- [ ] **Step 4: Create `blur_up.frag`**

Wider upsample for smoother blending.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_half_pixel;

void main() {
    vec2 o = niri_half_pixel * 4.0;
    vec4 sum = vec4(0.0);
    sum += texture2D(niri_input, v_coords + vec2(-o.x * 2.0, 0.0));
    sum += texture2D(niri_input, v_coords + vec2( o.x * 2.0, 0.0));
    sum += texture2D(niri_input, v_coords + vec2(0.0, -o.y * 2.0));
    sum += texture2D(niri_input, v_coords + vec2(0.0,  o.y * 2.0));
    sum += texture2D(niri_input, v_coords + vec2(-o.x,  o.y)) * 2.0;
    sum += texture2D(niri_input, v_coords + vec2( o.x,  o.y)) * 2.0;
    sum += texture2D(niri_input, v_coords + vec2(-o.x, -o.y)) * 2.0;
    sum += texture2D(niri_input, v_coords + vec2( o.x, -o.y)) * 2.0;
    gl_FragColor = sum / 12.0;
}
```

- [ ] **Step 5: Create `composite.frag`**

Edge highlights, specular bloom, chromatic fringing, environment reflection tint.

```glsl
#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0;

    // Chromatic fringing concentrated at borders.
    float chromatic_strength = smoothstep(0.2, 1.0, dist) * 0.004;
    vec2 dir = normalize(center + 0.001) * chromatic_strength;

    float r = texture2D(niri_input, v_coords + dir).r;
    float g = texture2D(niri_input, v_coords).g;
    float b = texture2D(niri_input, v_coords - dir).b;
    float a = texture2D(niri_input, v_coords).a;

    vec4 color = vec4(r, g, b, a);

    // Bright edge highlight: simulates light catching the glass rim.
    float edge_highlight = smoothstep(0.7, 1.0, dist) * 0.2;
    color.rgb += vec3(edge_highlight);

    // Specular bloom from top-left.
    vec2 light_dir = normalize(vec2(-0.5, -0.5));
    float specular = pow(max(0.0, dot(normalize(center), light_dir)), 16.0) * 0.12;
    color.rgb += vec3(specular);

    // Subtle environment reflection tint (cool blue).
    color.rgb += vec3(0.01, 0.02, 0.04) * smoothstep(0.4, 1.0, dist);

    gl_FragColor = color;
}
```

- [ ] **Step 6: Commit**

```bash
git add docs/wiki/examples/glass-liquid/
git commit -m "feat: add Liquid Glass (Apple-like) example shader"
```

---

### Task 10: Final verification

- [ ] **Step 1: Full build**

Run: `cargo build`
Expected: Clean build.

- [ ] **Step 2: Run all tests**

Run: `cargo test`
Expected: All tests pass.

- [ ] **Step 3: Verify example shaders parse correctly**

Write a quick inline test or manually verify the pipeline.kdl files are valid KDL by running:

```bash
ls -la docs/wiki/examples/glass-physics/
ls -la docs/wiki/examples/glass-liquid/
```

Expected: Both directories contain `pipeline.kdl` and the expected `.frag` files.

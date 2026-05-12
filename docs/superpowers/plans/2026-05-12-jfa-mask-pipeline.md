# JFA Mask Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tangled JFA mask pipeline with a clean five-stage pipeline producing an SDF+direction texture that uses multi-scale density gradients to give smooth interior curves while keeping boundary sharpness.

**Architecture:** Five GL fragment-shader stages in bbox space — binary mask → JFA init → JFA steps (ping-pong) → separable density blur (2px and 20px packed in R/G) → encode (combine JFA distance with blended density gradients). Rust orchestration in `src/render_helpers/blur.rs` is refactored from a single 300-line `render_jfa_mask` block into named per-stage functions over a named texture struct, eliminating the current `read_idx` / `write_idx` index dance.

**Tech Stack:** Rust + OpenGL ES 3.0 GLSL via the Smithay `GlesRenderer`. RGBA16F intermediate textures. No new dependencies.

**Validation:** This is a graphics task — there are no unit tests. After each task, build (`cargo build`) to catch compile errors. Final visual verification with the `overshifted3` pipeline on a multi-region window, optionally inspected via the `jfa-debug` pipeline.

---

## File Structure

- **Modify:** `src/render_helpers/blur.rs` — refactor `JfaPipeline`, add `JfaTextures` struct, split `render_jfa_mask` into named stages, add density blur program.
- **Rewrite:** `src/render_helpers/shaders/jfa_density_blur.frag` — currently a stub returning a constant; replace with a real separable box blur.
- **Rewrite:** `src/render_helpers/shaders/jfa_encode.frag` — drop the inline distance-blur hack; consume the precomputed density texture instead.
- **Delete:** `src/render_helpers/shaders/jfa_blur.frag` — dead, syntactically broken file.
- **Untouched:** `src/render_helpers/shaders/mask_binary.frag`, `jfa_init.frag`, `jfa_step.frag`, `mask.frag`, all `shaders/overshifted3/*.frag`, `shaders/jfa-debug/viz.frag`.

---

### Task 1: Rewrite `jfa_density_blur.frag` as a real separable box blur

**Files:**
- Modify: `src/render_helpers/shaders/jfa_density_blur.frag`

Goal: read the binary mask, perform a 1D box blur along one axis with two simultaneous radii, output `R = density_low`, `G = density_high`.

When `niri_axis == 0` it reads from the binary mask (R channel) and outputs the horizontal-blur intermediate. When `niri_axis == 1` it reads from the horizontal intermediate (R, G channels each separately blurred vertically) and outputs the final density.

- [ ] **Step 1: Replace the file contents**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform int niri_axis;           // 0 = horizontal, 1 = vertical
uniform int niri_radius_low;     // small radius (e.g. 2)
uniform int niri_radius_high;    // large radius (e.g. 20)

out vec4 frag_color;

// Must be >= max(niri_radius_low, niri_radius_high). 20px is the spec default
// for the high radius; 24 leaves a little headroom for experimentation.
const int MAX_R = 24;

void main() {
    vec2 uv = v_coords;
    vec2 step_uv = (niri_axis == 0)
        ? vec2(1.0 / niri_output_size.x, 0.0)
        : vec2(0.0, 1.0 / niri_output_size.y);

    float sum_low = 0.0;
    float sum_high = 0.0;
    float count_low = 0.0;
    float count_high = 0.0;

    for (int i = -MAX_R; i <= MAX_R; i++) {
        vec2 s_uv = clamp(uv + float(i) * step_uv, 0.0, 1.0);
        vec4 s = texture(niri_input, s_uv);

        // On the horizontal pass we read the binary mask: R holds the value
        // and both channels accumulate from R. On the vertical pass we read
        // the horizontal intermediate: R is already low-blurred horizontally,
        // G is high-blurred horizontally.
        float src_low  = (niri_axis == 0) ? s.r : s.r;
        float src_high = (niri_axis == 0) ? s.r : s.g;

        if (abs(i) <= niri_radius_low) {
            sum_low += src_low;
            count_low += 1.0;
        }
        if (abs(i) <= niri_radius_high) {
            sum_high += src_high;
            count_high += 1.0;
        }
    }

    float density_low = count_low > 0.0 ? sum_low / count_low : 0.0;
    float density_high = count_high > 0.0 ? sum_high / count_high : 0.0;

    frag_color = vec4(density_low, density_high, 0.0, 1.0);
}
```

- [ ] **Step 2: Compile-check by building**

Run: `cargo build` from `/home/calops/projects/niri-calops`
Expected: PASS (shader is loaded via `include_str!` at runtime, so build only verifies the surrounding Rust still compiles).

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_density_blur.frag
git commit -m "feat(shaders): real separable box blur for JFA density field"
```

---

### Task 2: Rewrite `jfa_encode.frag` to consume the density texture

**Files:**
- Modify: `src/render_helpers/shaders/jfa_encode.frag`

Drop the inline `blur_dist` / `dir_from_blurred_dist` helpers (they are the old hack). Read the JFA result on texture unit 0 and the density texture on a new sampler `niri_density` on texture unit 1. Compute gradients of `density_low` (R) and `density_high` (G) by central differences in pixel-space, blend by `smoothstep(0, niri_edge_threshold_px, distance)`, normalize, and pack into G/B.

- [ ] **Step 1: Replace the file contents**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;             // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_density;           // R = density_low, G = density_high
uniform vec2 niri_output_size;
uniform float niri_max_dist;              // normalization scale for the R channel
uniform float niri_edge_threshold_px;     // pixel distance at which blend reaches the high (smooth) gradient

out vec4 frag_color;

float dist_at(vec2 uv) {
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(niri_input, uv).rg;
    if (nearest.x < 0.0) return -1.0;
    return length(nearest - pixel);
}

void main() {
    vec2 uv = v_coords;
    float dc = dist_at(uv);

    if (dc < 0.0) {
        // Exterior sentinel — matches the analytical SDF path so renderers'
        // `mask < 0.001` early-out still triggers.
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float mask = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Central-difference gradient of each density channel, in pixel units.
    vec2 st = 1.0 / niri_output_size;
    vec4 dx = texture(niri_density, uv + vec2(st.x, 0.0))
            - texture(niri_density, uv - vec2(st.x, 0.0));
    vec4 dy = texture(niri_density, uv + vec2(0.0, st.y))
            - texture(niri_density, uv - vec2(0.0, st.y));

    // Density rises toward the interior, so the gradient points inward — exactly
    // what GB encodes for the renderers (`to_center` direction).
    vec2 grad_low = vec2(dx.r, dy.r);
    vec2 grad_high = vec2(dx.g, dy.g);

    float t = smoothstep(0.0, niri_edge_threshold_px, dc);
    vec2 dir = mix(grad_low, grad_high, t);

    float len = length(dir);
    vec2 norm_dir = len > 1e-6 ? dir / len : vec2(0.0);

    frag_color = vec4(mask, norm_dir.x * 0.5 + 0.5, norm_dir.y * 0.5 + 0.5, 1.0);
}
```

- [ ] **Step 2: Compile-check by building**

Run: `cargo build`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_encode.frag
git commit -m "feat(shaders): JFA encode reads precomputed density gradients"
```

---

### Task 3: Delete the dead `jfa_blur.frag`

**Files:**
- Delete: `src/render_helpers/shaders/jfa_blur.frag`

This file is unused, syntactically broken (unclosed brace, dead code), and grep confirms nothing references it.

- [ ] **Step 1: Confirm nothing references the file**

Run: `rg "jfa_blur" src/`
Expected: no matches (the file is never `include_str!`'d).

- [ ] **Step 2: Delete the file**

```bash
rm src/render_helpers/shaders/jfa_blur.frag
```

- [ ] **Step 3: Build to confirm nothing breaks**

Run: `cargo build`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A src/render_helpers/shaders/
git commit -m "chore(shaders): remove dead jfa_blur.frag"
```

---

### Task 4: Add `JfaDensityBlurProgram` struct, compile fn, and pipeline field

**Files:**
- Modify: `src/render_helpers/blur.rs`

Add a new program struct for the density blur shader, a compile function in the existing style, and wire it into `JfaPipeline`.

- [ ] **Step 1: Add the struct definition**

Insert after the existing `JfaEncodeProgram` definition (around `blur.rs:267`):

```rust
#[derive(Debug)]
struct JfaDensityBlurProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_axis: ffi::types::GLint,
    uniform_radius_low: ffi::types::GLint,
    uniform_radius_high: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

- [ ] **Step 2: Add a `density_prog` field to `JfaPipeline`**

Replace the existing `JfaPipeline` struct definition with:

```rust
#[derive(Debug)]
struct JfaPipeline {
    binary_prog: JfaBinaryProgram,
    init_prog: JfaInitProgram,
    step_prog: JfaStepProgram,
    density_prog: JfaDensityBlurProgram,
    encode_prog: JfaEncodeProgram,
}
```

- [ ] **Step 3: Add the compile function**

Insert after `compile_jfa_step` (around `blur.rs:333`):

```rust
unsafe fn compile_jfa_density_blur(gl: &ffi::Gles2) -> Result<JfaDensityBlurProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_density_blur.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaDensityBlurProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_axis: gl.GetUniformLocation(program, c"niri_axis".as_ptr()),
        uniform_radius_low: gl.GetUniformLocation(program, c"niri_radius_low".as_ptr()),
        uniform_radius_high: gl.GetUniformLocation(program, c"niri_radius_high".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}
```

- [ ] **Step 4: Add new uniform locations to `JfaEncodeProgram`**

Replace the existing `JfaEncodeProgram` struct with:

```rust
#[derive(Debug)]
struct JfaEncodeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_density: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
    uniform_edge_threshold_px: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

Then update `compile_jfa_encode` to read the new locations. Replace the function with:

```rust
unsafe fn compile_jfa_encode(gl: &ffi::Gles2) -> Result<JfaEncodeProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_encode.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaEncodeProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_density: gl.GetUniformLocation(program, c"niri_density".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_max_dist: gl.GetUniformLocation(program, c"niri_max_dist".as_ptr()),
        uniform_edge_threshold_px: gl.GetUniformLocation(
            program,
            c"niri_edge_threshold_px".as_ptr(),
        ),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}
```

- [ ] **Step 5: Wire `density_prog` into the pipeline construction**

In `render_jfa_mask` (around `blur.rs:791-797`), update the `JfaPipeline` construction:

```rust
Ok(JfaPipeline {
    binary_prog: compile_jfa_binary(gl)?,
    init_prog: compile_jfa_init(gl)?,
    step_prog: compile_jfa_step(gl)?,
    density_prog: compile_jfa_density_blur(gl)?,
    encode_prog: compile_jfa_encode(gl)?,
})
```

- [ ] **Step 6: Build**

Run: `cargo build`
Expected: PASS. (The old `render_jfa_mask` code still works because the encode program still has the uniform locations it uses; new uniforms are added but unused yet.)

- [ ] **Step 7: Commit**

```bash
git add src/render_helpers/blur.rs
git commit -m "feat(blur): add JFA density blur program and pipeline slot"
```

---

### Task 5: Introduce a named `JfaTextures` struct in place of the `Vec<GlesTexture>` index dance

**Files:**
- Modify: `src/render_helpers/blur.rs`

Replace `self.jfa_textures: Vec<GlesTexture>` with a struct of named slots. This is purely a data-structure refactor — keep behavior identical for now.

- [ ] **Step 1: Define the struct**

Insert after `JfaPipeline` (around `blur.rs:275`):

```rust
#[derive(Debug)]
struct JfaTextures {
    /// Binary mask of the subregion union.
    bin: GlesTexture,
    /// JFA ping-pong A (also reused as the final encoded output before blit).
    jfa_a: GlesTexture,
    /// JFA ping-pong B.
    jfa_b: GlesTexture,
    /// Density blur horizontal intermediate (R=low, G=high).
    density_h: GlesTexture,
    /// Density blur final (R=low, G=high).
    density: GlesTexture,
    /// Final encoded output (R=normalized SDF, GB=encoded direction).
    encoded: GlesTexture,
    /// Bbox size these textures were allocated for.
    size: Size<i32, Buffer>,
}
```

- [ ] **Step 2: Replace the `jfa_textures` field on `Blur`**

In the `Blur` struct (around `blur.rs:33-34`), replace:

```rust
    jfa_pipeline: Option<JfaPipeline>,
    jfa_textures: Vec<GlesTexture>,
```

with:

```rust
    jfa_pipeline: Option<JfaPipeline>,
    jfa_textures: Option<JfaTextures>,
```

And update the constructor in `Blur::new` (around `blur.rs:361-362`):

```rust
            jfa_pipeline: None,
            jfa_textures: None,
```

- [ ] **Step 3: Replace the allocation block in `render_custom`**

In `render_custom`, find the block currently around `blur.rs:537-543`:

```rust
                if bbw > 0 && bbh > 0 {
                    let bbox_size = Size::new(bbw, bbh);
                    self.jfa_textures.clear();
                    for _ in 0..4 {
                        let tex = renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?;
                        self.jfa_textures.push(tex);
                    }
                    Some((bbx, bby, bbw, bbh))
                } else {
                    None
                }
```

Replace with:

```rust
                if bbw > 0 && bbh > 0 {
                    let bbox_size = Size::new(bbw, bbh);
                    let need_alloc = match &self.jfa_textures {
                        Some(t) => t.size != bbox_size,
                        None => true,
                    };
                    if need_alloc {
                        let mk = || renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size);
                        self.jfa_textures = Some(JfaTextures {
                            bin: mk()?,
                            jfa_a: mk()?,
                            jfa_b: mk()?,
                            density_h: mk()?,
                            density: mk()?,
                            encoded: mk()?,
                            size: bbox_size,
                        });
                    }
                    Some((bbx, bby, bbw, bbh))
                } else {
                    None
                }
```

- [ ] **Step 4: Update the `render_jfa_mask` call site**

In `render_custom` (around `blur.rs:563-575`), replace:

```rust
                        render_jfa_mask(
                            gl,
                            options,
                            mask_tex_id,
                            bbx,
                            bby,
                            bbw,
                            bbh,
                            source_size.w,
                            source_size.h,
                            &mut self.jfa_pipeline,
                            &self.jfa_textures,
                        );
```

with:

```rust
                        let textures = self.jfa_textures.as_ref().unwrap();
                        render_jfa_mask(
                            gl,
                            options,
                            mask_tex_id,
                            bbx,
                            bby,
                            bbw,
                            bbh,
                            source_size.w,
                            source_size.h,
                            &mut self.jfa_pipeline,
                            textures,
                        );
```

- [ ] **Step 5: Update `render_jfa_mask`'s signature and stop using indices**

Change the function signature (around `blur.rs:761-773`):

```rust
fn render_jfa_mask(
    gl: &ffi::Gles2,
    options: &BlurOptions,
    mask_tex_id: ffi::types::GLuint,
    bbx: i32,
    bby: i32,
    bbw: i32,
    bbh: i32,
    source_w: i32,
    source_h: i32,
    jfa_pipeline: &mut Option<JfaPipeline>,
    textures: &JfaTextures,
) {
```

Inside the function body, replace every `tex[N]` reference with the corresponding named field. The mapping for the EXISTING (pre-density) pipeline is:

| Old | New |
|---|---|
| `tex[0]` (binary mask, also reused as encode output) | `textures.bin` for binary mask use; `textures.encoded` for encode-output use |
| `tex[1]` (JFA init dest, then JFA read start) | `textures.jfa_a` |
| `tex[2]` (JFA write) | `textures.jfa_b` |
| `tex[3]` (unused in current code) | not used yet |

And the JFA ping-pong's `read_idx` / `write_idx` index variables become explicit local `&GlesTexture` bindings that swap each iteration:

```rust
let mut read_tex = &textures.jfa_a;
let mut write_tex = &textures.jfa_b;
// ...inside the while loop, after the draw:
std::mem::swap(&mut read_tex, &mut write_tex);
```

The current code has `let tex = jfa_textures;` near the top — replace that and the entire `let mut read_idx = 1; let mut write_idx = 2;` block with:

```rust
let mut read_tex: &GlesTexture = &textures.jfa_a;
let mut write_tex: &GlesTexture = &textures.jfa_b;
```

Within the JFA loop, replace `tex[write_idx]` with `write_tex`, `tex[read_idx]` with `read_tex`. Replace `std::mem::swap(&mut read_idx, &mut write_idx)` with `std::mem::swap(&mut read_tex, &mut write_tex)`.

The encode pass currently reads `tex[read_idx]` and writes to `tex[0]` — keep behavior identical: read from `read_tex`, write to `textures.encoded`. The blit source becomes `textures.encoded`.

The binary-mask write target (after `compile_jfa_binary`) currently uses `tex[0].tex_id()` — change to `textures.bin.tex_id()`. The JFA init read uses `tex[0]` — change to `textures.bin`. The JFA init write uses `tex[1]` — change to `textures.jfa_a`.

- [ ] **Step 6: Build**

Run: `cargo build`
Expected: PASS.

- [ ] **Step 7: Visual smoke test (optional but recommended)**

Run niri with the existing `overshifted3` pipeline on a multi-region window. Visual output should be **identical** to before — this task is a pure data-structure refactor.

- [ ] **Step 8: Commit**

```bash
git add src/render_helpers/blur.rs
git commit -m "refactor(blur): name JFA textures, drop the index dance"
```

---

### Task 6: Wire the density blur and updated encode into `render_jfa_mask`

**Files:**
- Modify: `src/render_helpers/blur.rs`

Add the two density-blur passes (horizontal then vertical) between the JFA loop and the encode pass, and bind the density texture on TEXTURE1 for the encode pass. Also update `niri_max_dist` to be bbox-relative (`min(bbw, bbh) / 2`) and set `niri_edge_threshold_px`.

- [ ] **Step 1: Locate the insertion point**

In `render_jfa_mask`, find the comment block that currently reads:

```rust
        // tex[0] = binary mask (unchanged since init read from it)
        // tex[read_idx] = JFA distance field
        // tex[3], tex[4] = density blur destinations
        // tex[5] = temp for separable blur intermediate

        // Encode: reads JFA tex[read_idx], writes tex[0].
```

That comment block goes away. The new flow is: after the JFA `while step > 0` loop ends, run the density-blur horizontal pass, then vertical pass, then the encode pass.

- [ ] **Step 2: Insert the two density blur passes**

After the `while step > 0 { ... }` loop closes and BEFORE the encode pass setup, insert:

```rust
        // Density blur pass 1: horizontal. Reads textures.bin (binary mask),
        // writes textures.density_h with R = density_low (2px), G = density_high (20px).
        let radius_low: i32 = 2;
        let radius_high: i32 = 20;

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            textures.density_h.tex_id(),
            0,
        );

        gl.UseProgram(pipeline.density_prog.program);
        gl.Uniform1i(pipeline.density_prog.uniform_input, 0);
        gl.Uniform2f(
            pipeline.density_prog.uniform_output_size,
            bbw as f32,
            bbh as f32,
        );
        gl.Uniform1i(pipeline.density_prog.uniform_axis, 0);
        gl.Uniform1i(pipeline.density_prog.uniform_radius_low, radius_low);
        gl.Uniform1i(pipeline.density_prog.uniform_radius_high, radius_high);

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, textures.bin.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
        gl.EnableVertexAttribArray(pipeline.density_prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            pipeline.density_prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(pipeline.density_prog.attrib_vert as u32);

        // Density blur pass 2: vertical. Reads textures.density_h, writes textures.density.
        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            textures.density.tex_id(),
            0,
        );

        gl.Uniform1i(pipeline.density_prog.uniform_axis, 1);

        gl.BindTexture(ffi::TEXTURE_2D, textures.density_h.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
        gl.EnableVertexAttribArray(pipeline.density_prog.attrib_vert as u32);
        gl.VertexAttribPointer(
            pipeline.density_prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(pipeline.density_prog.attrib_vert as u32);

        // After the vertical pass, leave TEXTURE0 ready for the encode's JFA input.
```

- [ ] **Step 3: Update the encode pass setup**

The encode pass currently looks roughly like:

```rust
        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER, ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D, textures.encoded.tex_id(), 0,
        );

        gl.UseProgram(pipeline.encode_prog.program);
        gl.Uniform1i(pipeline.encode_prog.uniform_input, 0);
        gl.Uniform2f(pipeline.encode_prog.uniform_output_size, bbw as f32, bbh as f32);
        gl.Uniform1f(pipeline.encode_prog.uniform_max_dist, max_dim as f32 / 2.0);

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, read_tex.tex_id());
        // ...DrawArrays etc
```

Replace the encode pass setup with (binding the density texture to TEXTURE1, setting the new uniforms, switching to bbox-relative `niri_max_dist`):

```rust
        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER, ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D, textures.encoded.tex_id(), 0,
        );

        gl.UseProgram(pipeline.encode_prog.program);
        gl.Uniform1i(pipeline.encode_prog.uniform_input, 0);
        gl.Uniform1i(pipeline.encode_prog.uniform_density, 1);
        gl.Uniform2f(pipeline.encode_prog.uniform_output_size, bbw as f32, bbh as f32);

        // Bbox-relative normalization: deep interior reads near R=1.0 regardless
        // of bbox aspect.
        let max_dist = (std::cmp::min(bbw, bbh) as f32) / 2.0;
        gl.Uniform1f(pipeline.encode_prog.uniform_max_dist, max_dist);

        // Pixel distance over which the blend transitions from sharp (low-radius)
        // gradient to smooth (high-radius) gradient. Matches the high blur radius.
        gl.Uniform1f(pipeline.encode_prog.uniform_edge_threshold_px, 20.0);

        gl.Viewport(0, 0, bbw, bbh);

        // Bind density on TEXTURE1, JFA result on TEXTURE0.
        gl.ActiveTexture(ffi::TEXTURE1);
        gl.BindTexture(ffi::TEXTURE_2D, textures.density.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
        gl.ActiveTexture(ffi::TEXTURE0);
        gl.BindTexture(ffi::TEXTURE_2D, read_tex.tex_id());

        gl.EnableVertexAttribArray(pipeline.encode_prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            pipeline.encode_prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(pipeline.encode_prog.attrib_vert as u32);
```

(Note: the `let max_dim = max(bbw, bbh);` line earlier in the function is still used for the JFA step seed and stays as-is — `max_dim` and `max_dist` are now distinct.)

- [ ] **Step 4: Build**

Run: `cargo build`
Expected: PASS.

- [ ] **Step 5: Visual verification**

Run niri with the `overshifted3` custom-blur pipeline configured on a multi-region window (e.g. via the user's existing test setup). Expected behavior:
- Boundary remains sharp (glass-edge highlight from `glass.frag` still tracks the rect outline cleanly).
- Interior refraction shows smooth, credible curves rather than the previous bumpy gradient.
- No flicker, no NaNs (which would manifest as black squares or wild colors).

Optionally swap the pipeline to `jfa-debug` to visualize the encoded `R` (depth) and `GB` (direction). The R channel should ramp smoothly from 0 at edges to ~1 at the bbox center; the GB vector should point inward everywhere with a smooth angle field.

- [ ] **Step 6: Commit**

```bash
git add src/render_helpers/blur.rs
git commit -m "feat(blur): blend low/high density gradients in JFA encode"
```

---

### Task 7: Split `render_jfa_mask` into named stage functions

**Files:**
- Modify: `src/render_helpers/blur.rs`

`render_jfa_mask` is now ~250 lines. Split it into focused stage functions that take explicit input/output texture references. This is a readability refactor — no behavior change.

Target shape:

```rust
fn render_jfa_mask(
    gl: &ffi::Gles2,
    options: &BlurOptions,
    mask_tex_id: ffi::types::GLuint,
    bbx: i32, bby: i32, bbw: i32, bbh: i32,
    source_w: i32, source_h: i32,
    jfa_pipeline: &mut Option<JfaPipeline>,
    textures: &JfaTextures,
) {
    unsafe {
        clear_mask_texture(gl, mask_tex_id);
        if !ensure_pipeline(gl, jfa_pipeline) {
            return;
        }
        let pipeline = jfa_pipeline.as_ref().unwrap();

        let mut fbo = 0u32;
        gl.GenFramebuffers(1, &mut fbo);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbo);

        render_binary_mask(gl, &pipeline.binary_prog, options, &textures.bin,
            bbx, bby, bbw, bbh, source_w, source_h);
        jfa_init_pass(gl, &pipeline.init_prog, &textures.bin, &textures.jfa_a, bbw, bbh);
        let jfa_result = run_jfa_steps(gl, &pipeline.step_prog,
            &textures.jfa_a, &textures.jfa_b, bbw, bbh);
        compute_density(gl, &pipeline.density_prog,
            &textures.bin, &textures.density_h, &textures.density,
            bbw, bbh, 2, 20);
        encode_output(gl, &pipeline.encode_prog,
            jfa_result, &textures.density, &textures.encoded, bbw, bbh);
        blit_to_mask_texture(gl, &textures.encoded, mask_tex_id,
            bbw, bbh, source_w, source_h, options, bbx, bby);

        gl.DeleteFramebuffers(1, &mut fbo);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);

        // Set final mask sampler params.
        gl.BindTexture(ffi::TEXTURE_2D, mask_tex_id);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
    }
}
```

- [ ] **Step 1: Extract `clear_mask_texture`**

```rust
unsafe fn clear_mask_texture(gl: &ffi::Gles2, mask_tex_id: ffi::types::GLuint) {
    let mut clear_fbo = 0u32;
    gl.GenFramebuffers(1, &mut clear_fbo);
    gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, clear_fbo);
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        mask_tex_id,
        0,
    );
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);
    gl.DeleteFramebuffers(1, &mut clear_fbo);
}
```

- [ ] **Step 2: Extract `ensure_pipeline`**

```rust
unsafe fn ensure_pipeline(gl: &ffi::Gles2, jfa_pipeline: &mut Option<JfaPipeline>) -> bool {
    if jfa_pipeline.is_some() {
        return true;
    }
    match (|| -> Result<JfaPipeline, GlesError> {
        Ok(JfaPipeline {
            binary_prog: compile_jfa_binary(gl)?,
            init_prog: compile_jfa_init(gl)?,
            step_prog: compile_jfa_step(gl)?,
            density_prog: compile_jfa_density_blur(gl)?,
            encode_prog: compile_jfa_encode(gl)?,
        })
    })() {
        Ok(p) => {
            *jfa_pipeline = Some(p);
            true
        }
        Err(err) => {
            warn!("error compiling JFA shaders: {err:?}");
            false
        }
    }
}
```

- [ ] **Step 3: Extract `render_binary_mask`**

This stage owns the rects-in-px / bbox-clamp math currently inlined. Move it verbatim into a function. Signature:

```rust
unsafe fn render_binary_mask(
    gl: &ffi::Gles2,
    prog: &JfaBinaryProgram,
    options: &BlurOptions,
    dst: &GlesTexture,
    bbx: i32, bby: i32, bbw: i32, bbh: i32,
    source_w: i32, source_h: i32,
) { /* binds dst as COLOR_ATTACHMENT0, computes rects_px, draws */ }
```

Move the existing block that:
- Binds the binary-program output to `textures.bin` (now `dst`).
- Builds `rects_px` from `options.subregion_rects` and bbox.
- Uses `UseProgram(pipeline.binary_prog.program)`, sets uniforms, draws.

Set `dst`'s sampler params (NEAREST, CLAMP_TO_EDGE) at the end as the original code does.

The `blit_x1` / `blit_y1` / `blit_x2` / `blit_y2` calculation currently lives inside `render_jfa_mask` and is used by the final blit. Keep that computation in `render_jfa_mask` (or move it into `blit_to_mask_texture` — see Step 7).

- [ ] **Step 4: Extract `jfa_init_pass`**

```rust
unsafe fn jfa_init_pass(
    gl: &ffi::Gles2,
    prog: &JfaInitProgram,
    src: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32, bbh: i32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);

    gl.Viewport(0, 0, bbw, bbh);
    gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);

    gl.BindTexture(ffi::TEXTURE_2D, dst.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
}
```

- [ ] **Step 5: Extract `run_jfa_steps`**

```rust
unsafe fn run_jfa_steps<'a>(
    gl: &ffi::Gles2,
    prog: &JfaStepProgram,
    initial: &'a GlesTexture,
    scratch: &'a GlesTexture,
    bbw: i32, bbh: i32,
) -> &'a GlesTexture {
    let max_dim = std::cmp::max(bbw, bbh);
    let mut step = 1;
    while step * 2 <= max_dim {
        step *= 2;
    }

    let mut read_tex: &GlesTexture = initial;
    let mut write_tex: &GlesTexture = scratch;

    while step > 0 {
        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            write_tex.tex_id(),
            0,
        );

        gl.UseProgram(prog.program);
        gl.Uniform1i(prog.uniform_input, 0);
        gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
        gl.Uniform1i(prog.uniform_step, step);

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, read_tex.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

        gl.EnableVertexAttribArray(prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(prog.attrib_vert as u32);

        std::mem::swap(&mut read_tex, &mut write_tex);
        step /= 2;
    }

    // After the final swap, `read_tex` holds the most recent write.
    read_tex
}
```

- [ ] **Step 6: Extract `compute_density`**

```rust
unsafe fn compute_density(
    gl: &ffi::Gles2,
    prog: &JfaDensityBlurProgram,
    bin: &GlesTexture,
    h_intermediate: &GlesTexture,
    final_density: &GlesTexture,
    bbw: i32, bbh: i32,
    radius_low: i32,
    radius_high: i32,
) {
    let mut run_pass = |src: &GlesTexture, dst: &GlesTexture, axis: i32| {
        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            dst.tex_id(),
            0,
        );

        gl.UseProgram(prog.program);
        gl.Uniform1i(prog.uniform_input, 0);
        gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
        gl.Uniform1i(prog.uniform_axis, axis);
        gl.Uniform1i(prog.uniform_radius_low, radius_low);
        gl.Uniform1i(prog.uniform_radius_high, radius_high);

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

        gl.EnableVertexAttribArray(prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(prog.attrib_vert as u32);
    };

    run_pass(bin, h_intermediate, 0);
    run_pass(h_intermediate, final_density, 1);
}
```

Note: closures over `gl` with FFI use are fine here because the closure is invoked twice synchronously and never escapes. If lifetimes are awkward, inline the two calls instead.

- [ ] **Step 7: Extract `encode_output`**

```rust
unsafe fn encode_output(
    gl: &ffi::Gles2,
    prog: &JfaEncodeProgram,
    jfa: &GlesTexture,
    density: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32, bbh: i32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.Uniform1i(prog.uniform_density, 1);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Uniform1f(prog.uniform_max_dist, (std::cmp::min(bbw, bbh) as f32) / 2.0);
    gl.Uniform1f(prog.uniform_edge_threshold_px, 20.0);

    gl.Viewport(0, 0, bbw, bbh);

    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, density.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, jfa.tex_id());

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
}
```

- [ ] **Step 8: Extract `blit_to_mask_texture`**

```rust
unsafe fn blit_to_mask_texture(
    gl: &ffi::Gles2,
    src_encoded: &GlesTexture,
    mask_tex_id: ffi::types::GLuint,
    bbw: i32, bbh: i32,
    source_w: i32, source_h: i32,
    options: &BlurOptions,
    bbx: i32, bby: i32,
) {
    // Recompute the blit destination rect from rects in source pixels.
    let rects_raw_px: Vec<[f32; 4]> = options
        .subregion_rects
        .iter()
        .map(|r| {
            [
                r[0] * source_w as f32,
                r[1] * source_h as f32,
                r[2] * source_w as f32,
                r[3] * source_h as f32,
            ]
        })
        .collect();

    let blit_x1 = rects_raw_px
        .iter()
        .map(|r| r[0] as i32)
        .min()
        .unwrap_or(bbx + 1)
        .max(0);
    let blit_y1 = rects_raw_px
        .iter()
        .map(|r| r[1] as i32)
        .min()
        .unwrap_or(bby + 1)
        .max(0);
    let blit_x2 = rects_raw_px
        .iter()
        .map(|r| r[2] as i32)
        .max()
        .unwrap_or(bbx + bbw - 1)
        .min(source_w);
    let blit_y2 = rects_raw_px
        .iter()
        .map(|r| r[3] as i32)
        .max()
        .unwrap_or(bby + bbh - 1)
        .min(source_h);

    let mut read_fbo = 0u32;
    gl.GenFramebuffers(1, &mut read_fbo);
    gl.BindFramebuffer(ffi::READ_FRAMEBUFFER, read_fbo);
    gl.FramebufferTexture2D(
        ffi::READ_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        src_encoded.tex_id(),
        0,
    );

    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        mask_tex_id,
        0,
    );

    gl.BlitFramebuffer(
        1, 1, bbw - 1, bbh - 1,
        blit_x1, blit_y1, blit_x2, blit_y2,
        ffi::COLOR_BUFFER_BIT,
        ffi::LINEAR,
    );

    gl.DeleteFramebuffers(1, &mut read_fbo);
}
```

- [ ] **Step 9: Replace the body of `render_jfa_mask`**

Replace the whole body with the orchestration shown at the top of this task. Delete the old monolithic implementation.

- [ ] **Step 10: Build**

Run: `cargo build`
Expected: PASS. Resolve any borrow-checker complaints by adjusting lifetimes (the `run_jfa_steps` return-borrow is the most likely friction point; if so, return the index of the final read texture instead and have the caller pick the right reference).

- [ ] **Step 11: Visual verification**

Same as Task 6 Step 5 — output should be functionally identical to Task 6 (this is a pure refactor).

- [ ] **Step 12: Commit**

```bash
git add src/render_helpers/blur.rs
git commit -m "refactor(blur): split render_jfa_mask into named stage functions"
```

---

## Self-Review Notes

**Spec coverage:**
- Stage 1 (binary mask): unchanged — covered (left as-is in Tasks 5/7).
- Stage 2 (JFA init): unchanged — covered.
- Stage 3 (JFA steps): refactored in Tasks 5 and 7 but behavior unchanged — covered.
- Stage 4 (density blur, separable, R=low/G=high, tweakable radii): Task 1 (shader) + Task 4 (program struct) + Task 6 (wiring) + Task 7 (extraction).
- Stage 5 (encode with blended gradients, bbox-relative max_dist, smoothstep blend, edge threshold uniform): Task 2 (shader) + Task 4 (uniforms) + Task 6 (wiring) + Task 7 (extraction).
- Texture layout (6 named slots): Task 5 (struct + alloc).
- Rust refactor into named stages: Task 7.
- Dead `jfa_blur.frag` deletion: Task 3.

**Type consistency:** All program-struct field names (`uniform_density`, `uniform_edge_threshold_px`, `uniform_radius_low`, `uniform_radius_high`, `uniform_axis`) are used consistently across compile fns, struct definitions, and wiring code.

**No placeholders:** Each step contains the exact code or exact diff guidance needed. No "TBD" or "handle edge cases" stubs.

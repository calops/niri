# JFA Mipmap SDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed density-blur direction path with a mipmapped-SDF approach that produces a smooth, ridge-free interior vector field while preserving sharp boundary detail.

**Architecture:** After JFA produces nearest-exterior coords per pixel, bake the distance into a scalar texture, generate a mipmap chain on it (effectively a multi-scale blur, ~free on GPU), and in the encode shader sample the SDF at a per-pixel LoD proportional to depth (`log2(dc / lod_base)`). Far from boundary → high mip → smooth gradient (no medial-axis ridges). Near boundary → mip 0 → sharp local gradient (preserves details).

**Tech Stack:** Rust + OpenGL ES 3.0 GLSL via Smithay's `GlesRenderer`. RGBA16F textures throughout. No new dependencies.

**Validation:** Graphics task — no unit tests. After each task, `cargo check` to catch compile errors. Final visual verification with `overshifted3` (smooth refraction, sharp edges) and `jfa-debug` (smooth R ramp + ridge-free direction field).

---

## File Structure

- **Create:** `src/render_helpers/shaders/jfa_sdf_bake.frag` — single-pass shader that reads the JFA result and writes scalar normalized SDF to R.
- **Rewrite:** `src/render_helpers/shaders/jfa_encode.frag` — drop density-gradient logic, sample mipmapped SDF with `textureLod` for direction.
- **Delete:** `src/render_helpers/shaders/jfa_density_blur.frag` — replaced.
- **Modify:** `src/render_helpers/blur.rs` — swap `JfaDensityBlurProgram` for `JfaSdfBakeProgram`, update `JfaEncodeProgram` uniforms, swap `JfaTextures` fields (`density_h` / `density` → `sdf`), replace `compute_density` stage with `bake_sdf` + `generate_sdf_mipmaps`, update `encode_output` to bind mipmapped SDF.
- **Untouched:** `mask_binary.frag`, `jfa_init.frag`, `jfa_step.frag`, `mask.frag`, `shaders/overshifted3/*`, `shaders/jfa-debug/viz.frag`.

---

### Task 1: Create the SDF bake fragment shader

**Files:**
- Create: `src/render_helpers/shaders/jfa_sdf_bake.frag`

This shader reads the JFA result (RG = nearest exterior pixel coord) and writes the scalar normalized distance to the R channel of an RGBA16F texture. Exterior pixels (where `nearest.x < 0`) write 0.

- [ ] **Step 1: Create the file**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;   // JFA result
uniform vec2 niri_output_size;
uniform float niri_max_dist;    // normalization scale

out vec4 frag_color;

void main() {
    vec2 pixel = v_coords * niri_output_size;
    vec2 nearest = texture(niri_input, v_coords).rg;

    float d;
    if (nearest.x < 0.0) {
        // Sentinel for "never seen by JFA" — treat as exterior.
        d = 0.0;
    } else {
        d = length(nearest - pixel);
    }

    float sdf = clamp(d / niri_max_dist, 0.0, 1.0);
    frag_color = vec4(sdf, 0.0, 0.0, 1.0);
}
```

- [ ] **Step 2: Build-check**

Run: `cargo check`
Expected: PASS. (Shader loaded at runtime via `include_str!`; this just confirms surrounding Rust still builds.)

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_sdf_bake.frag
git commit -m "feat(shaders): bake JFA result into scalar SDF texture"
```

---

### Task 2: Rewrite the encode shader to use mipmapped SDF for direction

**Files:**
- Modify: `src/render_helpers/shaders/jfa_encode.frag`

Direction comes from the gradient of the mipmapped SDF sampled at a per-pixel LoD proportional to `log2(dc / lod_base)`. The R channel (mask) still comes from the JFA distance directly (sharp).

- [ ] **Step 1: Replace the file contents**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;             // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_sdf_mip;           // mipmapped scalar SDF (R channel)
uniform vec2 niri_output_size;
uniform float niri_max_dist;              // normalization scale for the R channel
uniform float niri_lod_base;              // px of depth per LoD step (e.g. 4.0)
uniform float niri_max_lod;               // clamp ceiling for LoD

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(niri_input, uv).rg;

    if (nearest.x < 0.0) {
        // Exterior sentinel — matches the analytical SDF path.
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float dc = length(nearest - pixel);
    float mask = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Per-pixel LoD: deeper interior → higher mip → smoother field.
    // Logarithmic scaling: lod = log2(max(dc, lod_base) / lod_base)
    // makes the blur radius grow proportionally to depth.
    float lod = log2(max(dc, niri_lod_base) / niri_lod_base);
    lod = clamp(lod, 0.0, niri_max_lod);

    // Central-difference gradient at the chosen LoD. Step is one mip-0 texel;
    // textureLod's linear filter handles the smoothing at the chosen level.
    vec2 st = 1.0 / niri_output_size;
    float r = textureLod(niri_sdf_mip, uv + vec2(st.x, 0.0), lod).r;
    float l = textureLod(niri_sdf_mip, uv - vec2(st.x, 0.0), lod).r;
    float t = textureLod(niri_sdf_mip, uv + vec2(0.0, st.y), lod).r;
    float b = textureLod(niri_sdf_mip, uv - vec2(0.0, st.y), lod).r;

    // Gradient points toward increasing SDF — i.e. inward.
    vec2 grad = vec2(r - l, t - b);
    float len = length(grad);
    vec2 dir = len > 1e-6 ? grad / len : vec2(0.0);

    frag_color = vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1.0);
}
```

- [ ] **Step 2: Build-check**

Run: `cargo check`
Expected: PASS. The shader is loaded at runtime; this only verifies Rust still compiles. (The old encode shader's uniforms `niri_density` and `niri_edge_threshold_px` will become `-1` locations from the Rust side until Task 5 updates `JfaEncodeProgram` — harmless during this intermediate state.)

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_encode.frag
git commit -m "feat(shaders): JFA encode reads mipmapped SDF for smooth direction"
```

---

### Task 3: Add `JfaSdfBakeProgram` struct and compile function

**Files:**
- Modify: `src/render_helpers/blur.rs`

Add the program struct and a compile function in the existing style.

- [ ] **Step 1: Add the struct definition**

Insert immediately before `struct JfaDensityBlurProgram` (around `blur.rs:260`):

```rust
#[derive(Debug)]
struct JfaSdfBakeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

- [ ] **Step 2: Add the compile function**

Insert immediately before `unsafe fn compile_jfa_density_blur` (around `blur.rs:347`):

```rust
unsafe fn compile_jfa_sdf_bake(gl: &ffi::Gles2) -> Result<JfaSdfBakeProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_sdf_bake.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaSdfBakeProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_max_dist: gl.GetUniformLocation(program, c"niri_max_dist".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}
```

- [ ] **Step 3: Build**

Run: `cargo check`
Expected: PASS with a `function is never used` warning on `compile_jfa_sdf_bake` (it's not wired up yet).

- [ ] **Step 4: Commit**

```bash
git add src/render_helpers/blur.rs
git commit -m "feat(blur): add JfaSdfBakeProgram struct and compile fn"
```

---

### Task 4: Update `JfaEncodeProgram` uniforms

**Files:**
- Modify: `src/render_helpers/blur.rs`

Replace the density-blur uniforms with the mipmap-LoD uniforms.

- [ ] **Step 1: Replace the struct definition**

Find this block (around `blur.rs:271-280`):

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

Replace with:

```rust
#[derive(Debug)]
struct JfaEncodeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_sdf_mip: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
    uniform_lod_base: ffi::types::GLint,
    uniform_max_lod: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

- [ ] **Step 2: Replace `compile_jfa_encode`**

Find the existing `compile_jfa_encode` (around `blur.rs:365-381`) and replace with:

```rust
unsafe fn compile_jfa_encode(gl: &ffi::Gles2) -> Result<JfaEncodeProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_encode.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaEncodeProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_sdf_mip: gl.GetUniformLocation(program, c"niri_sdf_mip".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_max_dist: gl.GetUniformLocation(program, c"niri_max_dist".as_ptr()),
        uniform_lod_base: gl.GetUniformLocation(program, c"niri_lod_base".as_ptr()),
        uniform_max_lod: gl.GetUniformLocation(program, c"niri_max_lod".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}
```

- [ ] **Step 3: Build**

Run: `cargo check`
Expected: COMPILE ERROR — the existing `encode_output` function references `prog.uniform_density` and `prog.uniform_edge_threshold_px` which no longer exist. This is expected; Task 7 fixes it. Continue.

Note: if the build error prevents `cargo check` from running at all, that's still fine — we're aware. Otherwise it'll print the specific errors.

- [ ] **Step 4: Do NOT commit yet**

This task leaves the tree in a non-building state. The next task fixes `JfaTextures` and `JfaPipeline`; Task 6 fixes the call sites. We commit after Task 7 once the tree builds again.

---

### Task 5: Update `JfaPipeline` and `JfaTextures`

**Files:**
- Modify: `src/render_helpers/blur.rs`

Swap `density_prog` for `sdf_bake_prog` in `JfaPipeline`; swap the density textures for a single `sdf` texture in `JfaTextures`.

- [ ] **Step 1: Update `JfaPipeline`**

Find this block (around `blur.rs:283-291`):

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

Replace with:

```rust
#[derive(Debug)]
struct JfaPipeline {
    binary_prog: JfaBinaryProgram,
    init_prog: JfaInitProgram,
    step_prog: JfaStepProgram,
    sdf_bake_prog: JfaSdfBakeProgram,
    encode_prog: JfaEncodeProgram,
}
```

- [ ] **Step 2: Update `JfaTextures`**

Find this block (around `blur.rs:292-308`):

```rust
#[derive(Debug)]
struct JfaTextures {
    /// Binary mask of the subregion union.
    bin: GlesTexture,
    /// JFA ping-pong A.
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

Replace with:

```rust
#[derive(Debug)]
struct JfaTextures {
    /// Binary mask of the subregion union.
    bin: GlesTexture,
    /// JFA ping-pong A.
    jfa_a: GlesTexture,
    /// JFA ping-pong B.
    jfa_b: GlesTexture,
    /// Scalar SDF (R channel), mipmapped for per-pixel LoD sampling in encode.
    sdf: GlesTexture,
    /// Final encoded output (R=normalized SDF, GB=encoded direction).
    encoded: GlesTexture,
    /// Bbox size these textures were allocated for.
    size: Size<i32, Buffer>,
}
```

- [ ] **Step 3: Update the texture allocation block in `render_custom`**

Find this block (around `blur.rs:573-585`):

```rust
                    if need_alloc {
                        self.jfa_textures = Some(JfaTextures {
                            bin: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_a: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_b: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            density_h: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            density: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            encoded: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            size: bbox_size,
                        });
                    }
```

Replace with:

```rust
                    if need_alloc {
                        self.jfa_textures = Some(JfaTextures {
                            bin: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_a: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_b: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            sdf: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            encoded: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            size: bbox_size,
                        });
                    }
```

- [ ] **Step 4: Update `ensure_jfa_pipeline`**

Find this block (around `blur.rs:868-882`):

```rust
unsafe fn ensure_jfa_pipeline(gl: &ffi::Gles2, jfa_pipeline: &mut Option<JfaPipeline>) -> bool {
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
```

Replace the `density_prog:` line with `sdf_bake_prog:`:

```rust
unsafe fn ensure_jfa_pipeline(gl: &ffi::Gles2, jfa_pipeline: &mut Option<JfaPipeline>) -> bool {
    if jfa_pipeline.is_some() {
        return true;
    }
    match (|| -> Result<JfaPipeline, GlesError> {
        Ok(JfaPipeline {
            binary_prog: compile_jfa_binary(gl)?,
            init_prog: compile_jfa_init(gl)?,
            step_prog: compile_jfa_step(gl)?,
            sdf_bake_prog: compile_jfa_sdf_bake(gl)?,
            encode_prog: compile_jfa_encode(gl)?,
        })
    })() {
```

- [ ] **Step 5: Do NOT commit yet**

Still in mid-refactor. Tree won't build yet — the `compute_density` stage call in `render_jfa_mask` still references `&textures.density_h`, `&textures.density`, and `&pipeline.density_prog` which no longer exist.

---

### Task 6: Add `bake_sdf` and `generate_sdf_mipmaps` stages

**Files:**
- Modify: `src/render_helpers/blur.rs`

Replace the `compute_density` stage function with two new ones: `bake_sdf` (renders scalar SDF) and `generate_sdf_mipmaps` (calls `glGenerateMipmap` with proper sampler params).

- [ ] **Step 1: Remove `compute_density`**

Delete the entire `compute_density` function (around `blur.rs:1080-1130`).

- [ ] **Step 2: Add `bake_sdf`**

Insert in its place:

```rust
// Bakes the JFA result into a scalar normalized SDF texture (R channel).
// dc / max_dist clamped to [0, 1]. Exterior pixels get 0.
unsafe fn bake_sdf(
    gl: &ffi::Gles2,
    prog: &JfaSdfBakeProgram,
    jfa: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32,
    bbh: i32,
    max_dist: f32,
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
    gl.Uniform1f(prog.uniform_max_dist, max_dist);

    gl.Viewport(0, 0, bbw, bbh);
    gl.BindTexture(ffi::TEXTURE_2D, jfa.tex_id());
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
}
```

- [ ] **Step 3: Add `generate_sdf_mipmaps`**

Insert immediately after `bake_sdf`:

```rust
// Generates the full mipmap chain for the SDF texture. Smithay creates the
// texture with only level 0; glGenerateMipmap lazily allocates and fills the
// rest. Min-filter must be set to a mipmap variant before generation so the
// driver knows to build the chain.
unsafe fn generate_sdf_mipmaps(gl: &ffi::Gles2, sdf: &GlesTexture) {
    gl.BindTexture(ffi::TEXTURE_2D, sdf.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::LINEAR_MIPMAP_LINEAR as i32,
    );
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_WRAP_S,
        ffi::CLAMP_TO_EDGE as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_WRAP_T,
        ffi::CLAMP_TO_EDGE as i32,
    );
    gl.GenerateMipmap(ffi::TEXTURE_2D);
}
```

- [ ] **Step 4: Do NOT commit yet**

`render_jfa_mask` still calls the deleted `compute_density`. The next task fixes the orchestration and call sites.

---

### Task 7: Wire the new stages into `render_jfa_mask` and update `encode_output`

**Files:**
- Modify: `src/render_helpers/blur.rs`

Replace the `compute_density` call with `bake_sdf` + `generate_sdf_mipmaps`. Update `encode_output` to bind the mipmapped SDF on TEXTURE1 with mipmap-aware sampler params and set the new uniforms.

- [ ] **Step 1: Replace the `compute_density` call in `render_jfa_mask`**

Find this block (around `blur.rs:876-886`):

```rust
        compute_density(
            gl,
            &pipeline.density_prog,
            jfa_result,
            &textures.density_h,
            &textures.density,
            bbw,
            bbh,
            2,
            20,
        );
```

Replace with:

```rust
        // Compute the normalization scale once — used both by the SDF bake
        // (so the baked texture is in [0, 1]) and by encode_output (so the
        // mask R channel uses the same scale).
        let max_dist = (max(1, min(bbw, bbh)) as f32) / 2.0;

        bake_sdf(
            gl,
            &pipeline.sdf_bake_prog,
            jfa_result,
            &textures.sdf,
            bbw,
            bbh,
            max_dist,
        );
        generate_sdf_mipmaps(gl, &textures.sdf);
```

- [ ] **Step 2: Update the `encode_output` call**

Find this block (around `blur.rs:887-895`):

```rust
        encode_output(
            gl,
            &pipeline.encode_prog,
            jfa_result,
            &textures.density,
            &textures.encoded,
            bbw,
            bbh,
        );
```

Replace with:

```rust
        encode_output(
            gl,
            &pipeline.encode_prog,
            jfa_result,
            &textures.sdf,
            &textures.encoded,
            bbw,
            bbh,
            max_dist,
        );
```

- [ ] **Step 3: Update `encode_output` signature and body**

Find the existing `encode_output` (around `blur.rs:1132-1180`) and replace with:

```rust
unsafe fn encode_output(
    gl: &ffi::Gles2,
    prog: &JfaEncodeProgram,
    jfa: &GlesTexture,
    sdf_mip: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32,
    bbh: i32,
    max_dist: f32,
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
    gl.Uniform1i(prog.uniform_sdf_mip, 1);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Uniform1f(prog.uniform_max_dist, max_dist);

    // LoD scaling: a pixel at depth `2^k * lod_base` samples mip k. With
    // lod_base = 4, depths 4 / 8 / 16 / 32 / 64 / 128 px map to mip 0 / 1 / 2
    // / 3 / 4 / 5. Tunable in this one spot.
    let lod_base: f32 = 4.0;
    let mip_count = (max(bbw, bbh) as f32).log2().floor();
    gl.Uniform1f(prog.uniform_lod_base, lod_base);
    gl.Uniform1f(prog.uniform_max_lod, mip_count);

    gl.Viewport(0, 0, bbw, bbh);

    // Bind the mipmapped SDF on TEXTURE1 with mipmap-aware sampler params.
    // generate_sdf_mipmaps already set these, but re-asserting them costs
    // nothing and protects against accidental state mutation between passes.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, sdf_mip.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::LINEAR_MIPMAP_LINEAR as i32,
    );
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_WRAP_S,
        ffi::CLAMP_TO_EDGE as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_WRAP_T,
        ffi::CLAMP_TO_EDGE as i32,
    );
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

- [ ] **Step 4: Build**

Run: `cargo check`
Expected: PASS. Resolve any compile errors. The most likely friction is unused `min` import or similar — adjust as needed.

- [ ] **Step 5: Commit (consolidates Tasks 4–7 into one logical change)**

```bash
git add src/render_helpers/blur.rs
git commit -m "feat(blur): replace density blur with mipmapped SDF for direction"
```

---

### Task 8: Delete the dead density-blur shader and program

**Files:**
- Delete: `src/render_helpers/shaders/jfa_density_blur.frag`
- Modify: `src/render_helpers/blur.rs` (remove `JfaDensityBlurProgram` + `compile_jfa_density_blur`)

After Task 7, neither the struct nor the shader is referenced. Clean them up.

- [ ] **Step 1: Confirm nothing references the shader**

Run: `rg "jfa_density_blur|JfaDensityBlurProgram|compile_jfa_density_blur" src/`
Expected: matches only the struct definition and the compile fn in `blur.rs`, plus the shader filename. Nothing in `render_jfa_mask` or pipeline construction.

- [ ] **Step 2: Delete the shader file**

```bash
rm src/render_helpers/shaders/jfa_density_blur.frag
```

- [ ] **Step 3: Remove `JfaDensityBlurProgram`**

Find this block in `blur.rs` (around `blur.rs:260-269`):

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

Delete it entirely.

- [ ] **Step 4: Remove `compile_jfa_density_blur`**

Find this function in `blur.rs` (around `blur.rs:347-362`):

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

Delete it entirely.

- [ ] **Step 5: Build**

Run: `cargo check`
Expected: PASS, clean (no warnings about unused items).

- [ ] **Step 6: Commit**

```bash
git add -A src/render_helpers/
git commit -m "chore(blur): remove dead density blur path"
```

---

### Task 9: Visual verification

**Files:**
- None modified.

This is the validation gate. The user runs niri and inspects the output.

- [ ] **Step 1: Verify with `overshifted3` pipeline**

Configure niri's `custom_blur` to point at the `overshifted3/` pipeline directory and run niri. Look at a window with subregion-based blur. Expected:
- Smooth refraction throughout the interior, no visible seams.
- Sharp directional response at boundaries — corners, edges, and any thin features all preserve their shape.
- A plain rectangle should look qualitatively similar to the analytical-SDF full-window case (smooth radial-ish field).

- [ ] **Step 2: Verify with `jfa-debug` pipeline**

Swap the custom-blur pipeline to `jfa-debug/` and look at the same window. Expected:
- R channel ramps smoothly from 0 at the boundary to ~1 at the deepest interior point.
- GB direction field is smooth across the entire interior — no Voronoi-style ridges along diagonals.
- Near complex boundary features (notches, thin appendages, joints between rectangles), the direction snaps cleanly to the local perpendicular.

- [ ] **Step 3: If artifacts remain**

The two most likely tuning knobs:
- `lod_base` in `encode_output` (currently `4.0`). Raise to slow down the blur growth, lower to accelerate it.
- Mipmap quality. If deep mip levels look blocky, switch the SDF format from `Abgr16161616f` to manual `glTexImage2D(GL_R16F, ...)` for higher per-channel precision — but only if needed.

This step is purely about decision-making; no automatic verification.

- [ ] **Step 4: No commit needed for verification.**

---

## Self-Review Notes

**Spec coverage:**
- Pipeline stage 1 (binary mask): unchanged — covered (not touched).
- Pipeline stage 2 (JFA init): unchanged — covered.
- Pipeline stage 3 (JFA steps): unchanged — covered.
- Pipeline stage 4 (SDF bake): Task 1 (shader) + Task 3 (program struct + compile) + Task 7 (wiring).
- Pipeline stage 5 (mipmap generation): Task 6 (`generate_sdf_mipmaps`) + Task 7 (call).
- Pipeline stage 6 (encode with mipmapped SDF): Task 2 (shader) + Task 4 (uniforms) + Task 7 (encode_output rewrite).
- Texture layout (5 named slots, `sdf` replaces `density_h`/`density`): Task 5.
- Dead code removal: Task 8.

**Type consistency:**
- `JfaSdfBakeProgram` field names (`uniform_input`, `uniform_output_size`, `uniform_max_dist`) consistent across struct def (Task 3), compile fn (Task 3), and call site `bake_sdf` (Task 6).
- `JfaEncodeProgram` field names (`uniform_input`, `uniform_sdf_mip`, `uniform_output_size`, `uniform_max_dist`, `uniform_lod_base`, `uniform_max_lod`) consistent across struct def (Task 4), compile fn (Task 4), and `encode_output` (Task 7).
- `JfaTextures.sdf` consistently named.

**Tasks 4-7 leave the tree non-building intentionally.** This is called out and consolidates into one commit at the end of Task 7. The reason: these changes are tightly coupled — the `JfaTextures` field names, `JfaPipeline` field names, and the `compute_density` → `bake_sdf` switch must all happen together for the tree to compile. Splitting them would require temporary shims.

**No placeholders detected on review.**

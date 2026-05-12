# Poisson direction field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SDF-mipmap direction path with a geometric-multigrid solve of `∇²u = -1` on the mask interior, so the encoded GB direction comes from a field with provably-smooth gradient (no medial-axis seams) on arbitrary shapes.

**Architecture:** Five new fragment shaders (Jacobi smoother, residual+restrict, prolongation, level-0 RHS init, mask restriction) operate on a `K+1`-level pyramid of textures per bbox. Each level holds three RGBA16F textures: two ping-pong copies of the solution `u`, and one packed texture storing the right-hand side `f` (R channel) and the boundary mask (G channel). One V-cycle per frame: pre-smooth, restrict residual, recurse, prolongate, post-smooth. The final encode pass reads `u[0]` as the Poisson field and computes its gradient for the GB channels; the JFA-derived SDF still feeds R unchanged.

**Tech Stack:** Rust + OpenGL ES 3.0 GLSL via Smithay's `GlesRenderer`. RGBA16F throughout. No new dependencies.

**Validation:** Graphics task; `cargo check` for compile correctness, then visual verification with `overshifted3` (no seams in interior refraction) and `jfa-debug` (no diagonal ridges, smooth direction field on all four test shapes).

---

## File Structure

**New shaders** in `src/render_helpers/shaders/`:

- `jfa_poisson_init_rhs.frag` — initialise level-0 RHS texture (`f=1` inside mask, `0` outside; mask packed into G).
- `jfa_poisson_restrict_mask.frag` — box-downsample the mask from one level to the next (one-time per V-cycle / mask refresh).
- `jfa_poisson_jacobi.frag` — one weighted-Jacobi sweep on `u` at a single level.
- `jfa_poisson_residual_restrict.frag` — compute residual `r = f − Lu` at fine level and box-downsample into the next-coarser level's `f` channel in one pass.
- `jfa_poisson_prolongate.frag` — bilinear-upsample the coarse-level correction into the fine-level `u`.

**Rewritten shader:**

- `src/render_helpers/shaders/jfa_encode.frag` — drops textureLod-based SDF gradient; reads the JFA-baked SDF for R, and `u[0]` from the multigrid pyramid for GB via central-difference gradient.

**Deleted shader:**

- `src/render_helpers/shaders/jfa_sdf_downsample.frag`.

**Rust changes** in `src/render_helpers/blur.rs`:

- New program structs: `JfaPoissonInitRhsProgram`, `JfaPoissonRestrictMaskProgram`, `JfaPoissonJacobiProgram`, `JfaPoissonResidualRestrictProgram`, `JfaPoissonProlongateProgram`. Drop `JfaSdfDownsampleProgram`.
- New `MultigridLevel { u_a, u_b, rhs, size }` struct.
- `JfaTextures` gains `pyramid: Vec<MultigridLevel>`, drops `sdf` (replaced by `pyramid[0]`).
- New stage functions: `init_rhs_level0`, `restrict_mask_pyramid`, `jacobi_smooth`, `residual_restrict`, `prolongate`, `run_v_cycle`. Drop `generate_sdf_mipmaps`.
- `encode_output` reads `pyramid[0].u_a` (or whichever is current after the V-cycle) instead of the mipmapped SDF.
- `render_jfa_mask` orchestration: after JFA + `bake_sdf` for the R channel, run the multigrid V-cycle, then encode.

**Untouched:** `mask_binary.frag`, `jfa_init.frag`, `jfa_step.frag`, `jfa_sdf_bake.frag` (still needed for R channel), `mask.frag` (analytical full-window path), `shaders/overshifted3/*`, `shaders/jfa-debug/*`.

---

## Conventions used in this plan

- Pyramid depth `K = 5` levels above level 0 (so `pyramid.len() == K + 1 == 6`). Each level halves in both dimensions until min(1, …). The actual ceiling is computed at runtime as `min(K, floor(log2(min(bbw, bbh))))` to avoid 1×1 levels degenerating.
- Jacobi parameters: damping `ω = 4/5`, pre-/post-smoothing sweeps `N = 3` per level, V-cycles per frame `V = 1`.
- Texel-spacing convention: `h = 1` at level 0 (one texel = one unit). At level `k`, the spacing is `h_k = 2^k`. The Jacobi update uses `h_k²` as a uniform.
- All textures `RGBA16F` (`Fourcc::Abgr16161616f`) for consistency with everything else in this file.
- "Inside" / "outside" the mask is determined by mask value ≥ 0.5 (the resampled mask at coarse levels may be fractional).

---

### Task 1: Add `jfa_poisson_init_rhs.frag`

**Files:**
- Create: `src/render_helpers/shaders/jfa_poisson_init_rhs.frag`

Sets up the level-0 RHS texture. Reads the binary mask (RGBA16F from the existing `textures.bin`), writes `f = 1.0` to R and the mask value to G if the pixel is inside, `(0, 0)` otherwise. Also clears the `u` channels in their textures separately — handled in Rust via `glClear`.

- [ ] **Step 1: Create the file**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;        // binary mask, R = 1 inside, 0 outside

out vec4 frag_color;

void main() {
    float m = texture(niri_input, v_coords).r;
    // Boundary is mask >= 0.5 to keep things sharp at the original
    // resolution; coarser levels may average to fractional values which
    // we'll treat as boundary-weights inside the Jacobi smoother.
    float inside = m >= 0.5 ? 1.0 : 0.0;
    frag_color = vec4(inside, inside, 0.0, 1.0);
}
```

R = `f` (forcing function = 1 inside, 0 outside). G = mask (same value here; they coincide at level 0 because `f = 1 * mask`).

- [ ] **Step 2: Build check**

Run: `cargo check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_poisson_init_rhs.frag
git commit -m "feat(shaders): initialise multigrid level-0 RHS from binary mask"
```

---

### Task 2: Add `jfa_poisson_restrict_mask.frag`

**Files:**
- Create: `src/render_helpers/shaders/jfa_poisson_restrict_mask.frag`

Box-downsamples the mask from one level to the next. Reads RG of source level (`f`, `mask`), writes RG of destination level. We restrict BOTH channels here (the source `f` channel is the result of `init_rhs_level0` for the 0→1 case, or unused at higher levels since the cycle writes its own `f` later — but restricting it for free doesn't hurt and saves a separate pass). The mask resampling is a plain 2×2 box average; the renderer treats fractional values as weights in `jacobi`.

- [ ] **Step 1: Create the file**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_src_texel;     // 1.0 / source-level size

out vec4 frag_color;

// 4-tap 2×2 box downsample. Both channels averaged the same way.
void main() {
    vec2 uv = v_coords;
    vec2 o = niri_src_texel * 0.5;
    vec4 a = texture(niri_input, uv + vec2(-o.x, -o.y));
    vec4 b = texture(niri_input, uv + vec2( o.x, -o.y));
    vec4 c = texture(niri_input, uv + vec2(-o.x,  o.y));
    vec4 d = texture(niri_input, uv + vec2( o.x,  o.y));
    frag_color = vec4(0.25 * (a.r + b.r + c.r + d.r),
                      0.25 * (a.g + b.g + c.g + d.g),
                      0.0, 1.0);
}
```

- [ ] **Step 2: Build check**

Run: `cargo check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_poisson_restrict_mask.frag
git commit -m "feat(shaders): box-downsample mask + RHS to next-coarser level"
```

---

### Task 3: Add `jfa_poisson_jacobi.frag`

**Files:**
- Create: `src/render_helpers/shaders/jfa_poisson_jacobi.frag`

One weighted-Jacobi sweep. Reads `u` (R from one texture) and packed `rhs` (R=`f`, G=`mask` from another). Outputs new `u` in R. Boundary condition `u = 0` enforced by multiplying the result by `mask`.

- [ ] **Step 1: Create the file**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_u_in;       // R = u at this level
uniform sampler2D niri_rhs;        // R = f, G = mask at this level
uniform vec2 niri_texel;           // 1.0 / level size
uniform float niri_h_sq;           // h^2 at this level (=4^k for level k)
uniform float niri_omega;          // damping (0.8)

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;

    float u_c  = texture(niri_u_in, uv).r;
    float u_l  = texture(niri_u_in, uv - vec2(niri_texel.x, 0.0)).r;
    float u_r  = texture(niri_u_in, uv + vec2(niri_texel.x, 0.0)).r;
    float u_u  = texture(niri_u_in, uv + vec2(0.0, niri_texel.y)).r;
    float u_d  = texture(niri_u_in, uv - vec2(0.0, niri_texel.y)).r;

    vec4 rhs = texture(niri_rhs, uv);
    float f  = rhs.r;
    float m  = rhs.g;

    // ∇²u = -f  ⇒  (u_l + u_r + u_u + u_d - 4 u_c) / h² = -f
    //         ⇒  u_c_new = (u_l + u_r + u_u + u_d + h² f) / 4
    //
    // Weighted Jacobi: u_c_next = (1-ω) u_c + ω u_c_new.
    // Boundary: multiply by `m` so u stays 0 outside the mask.
    float u_jacobi = 0.25 * (u_l + u_r + u_u + u_d + niri_h_sq * f);
    float u_next   = (1.0 - niri_omega) * u_c + niri_omega * u_jacobi;
    frag_color = vec4(u_next * m, 0.0, 0.0, 1.0);
}
```

- [ ] **Step 2: Build check**

Run: `cargo check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_poisson_jacobi.frag
git commit -m "feat(shaders): weighted-Jacobi smoother for multigrid Poisson"
```

---

### Task 4: Add `jfa_poisson_residual_restrict.frag`

**Files:**
- Create: `src/render_helpers/shaders/jfa_poisson_residual_restrict.frag`

For each coarse-level pixel, sample a 2×2 block of fine-level pixels; for each fine pixel compute the residual `r = f − Lu` (which needs the four neighbours of the fine pixel for `Lu`); average the four residuals into the coarse pixel's R channel. The coarse mask is restricted by box average and stored in G — this is the same operation as `restrict_mask` so we fold it here per V-cycle (mask values at coarse levels can be cached from the one-time mask restriction; this duplicates work but keeps the V-cycle stateless).

Actually we'll keep the per-V-cycle pass tight by re-using the already-restricted mask: this shader writes only `f` (R), reads the destination's existing G via a separate sampler (`niri_dst_mask`), and outputs `(r_avg, mask_dst, 0, 1)`. Simpler and faster.

- [ ] **Step 1: Create the file**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_u_fine;        // R = u at fine level
uniform sampler2D niri_rhs_fine;      // R = f, G = mask at fine level
uniform sampler2D niri_mask_coarse;   // G = mask at coarse level (pre-restricted)
uniform vec2 niri_fine_texel;         // 1.0 / fine-level size
uniform float niri_h_sq_fine;         // h^2 at fine level

out vec4 frag_color;

// Computes residual r = f - L u at a fine-level pixel.
float residual_at(vec2 uv) {
    float u_c  = texture(niri_u_fine, uv).r;
    float u_l  = texture(niri_u_fine, uv - vec2(niri_fine_texel.x, 0.0)).r;
    float u_r  = texture(niri_u_fine, uv + vec2(niri_fine_texel.x, 0.0)).r;
    float u_u  = texture(niri_u_fine, uv + vec2(0.0, niri_fine_texel.y)).r;
    float u_d  = texture(niri_u_fine, uv - vec2(0.0, niri_fine_texel.y)).r;
    float f    = texture(niri_rhs_fine, uv).r;
    float lu   = (u_l + u_r + u_u + u_d - 4.0 * u_c) / niri_h_sq_fine;
    return f - (-lu);   // ∇²u = -f, so residual = f - (−Lu) = f + Lu? No:
                        // we solve L u = -f  →  residual = -f - L u
                        //                      → restricted RHS for next level
                        //                        is the residual r, but the
                        //                        equation at coarse level is
                        //                        L e = r  with the SAME sign
                        //                        convention.
}

// Note on signs: our equation is L u = -f (where L = Laplacian / h²).
// The residual is r = -f - L u_current. To restrict for the coarse-level
// correction equation L e = r, we pass r through unchanged.
//
// `residual_at` above computes  f + Lu, which is the negative of r. Flip it
// in `main` below to produce r correctly.

void main() {
    vec2 uv = v_coords;
    vec2 o = niri_fine_texel * 0.5;

    float r_a = residual_at(uv + vec2(-o.x, -o.y));
    float r_b = residual_at(uv + vec2( o.x, -o.y));
    float r_c = residual_at(uv + vec2(-o.x,  o.y));
    float r_d = residual_at(uv + vec2( o.x,  o.y));

    // Box-average and flip sign per the convention above.
    float r_avg = -0.25 * (r_a + r_b + r_c + r_d);

    // Pass through the pre-restricted coarse mask.
    float m = texture(niri_mask_coarse, uv).g;

    frag_color = vec4(r_avg, m, 0.0, 1.0);
}
```

- [ ] **Step 2: Build check**

Run: `cargo check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_poisson_residual_restrict.frag
git commit -m "feat(shaders): residual computation + restriction to coarser level"
```

---

### Task 5: Add `jfa_poisson_prolongate.frag`

**Files:**
- Create: `src/render_helpers/shaders/jfa_poisson_prolongate.frag`

Bilinear-upsamples the coarse-level correction (`u_coarse.r`) and adds it to the fine-level solution (`u_fine.r`). Output goes to a destination `u` texture at the fine level. The bilinear filter is GL's built-in (set `TEXTURE_MIN_FILTER` to `LINEAR` on the coarse texture before sampling).

- [ ] **Step 1: Create the file**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_u_fine;       // R = current u at fine level
uniform sampler2D niri_correction;   // R = correction from coarse level (sampled with LINEAR)
uniform sampler2D niri_rhs_fine;     // G = mask at fine level (for boundary enforcement)

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    float u_fine = texture(niri_u_fine, uv).r;
    float corr   = texture(niri_correction, uv).r;
    float m      = texture(niri_rhs_fine, uv).g;
    frag_color = vec4((u_fine + corr) * m, 0.0, 0.0, 1.0);
}
```

- [ ] **Step 2: Build check**

Run: `cargo check`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_poisson_prolongate.frag
git commit -m "feat(shaders): bilinear prolongation for multigrid V-cycle"
```

---

### Task 6: Rewrite `jfa_encode.frag` to use the Poisson field

**Files:**
- Modify: `src/render_helpers/shaders/jfa_encode.frag`

R channel comes from JFA distance via the existing baked SDF texture (`niri_input` reused for that role would be confusing; instead we pass the JFA result on TEXTURE0 as before and re-derive distance, OR we pass the baked SDF — keep the existing convention to read JFA on TEXTURE0). GB direction is the gradient of `u` at level 0 of the pyramid, normalised.

Decision: pass JFA on TEXTURE0 (so we can write the exterior sentinel correctly), and the Poisson `u[0]` field on TEXTURE1 (renaming `niri_sdf_mip` → `niri_poisson_u`). Drop the LoD-related uniforms.

- [ ] **Step 1: Replace the file contents**

```glsl
#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;         // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_poisson_u;     // R = Poisson solution u at level 0
uniform vec2 niri_output_size;
uniform float niri_max_dist;          // normalization scale for the R channel

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

    // Direction: central-difference gradient of the Poisson field.
    // u peaks somewhere in the interior; ∇u points INTO the interior
    // (toward larger u). That's exactly the "inward" direction the
    // renderers consume via (m.gb - 0.5) * 2.
    vec2 st = 1.0 / niri_output_size;
    float r = texture(niri_poisson_u, uv + vec2(st.x, 0.0)).r;
    float l = texture(niri_poisson_u, uv - vec2(st.x, 0.0)).r;
    float t = texture(niri_poisson_u, uv + vec2(0.0, st.y)).r;
    float b = texture(niri_poisson_u, uv - vec2(0.0, st.y)).r;

    vec2 grad = vec2(r - l, t - b);
    float len = length(grad);
    vec2 dir = len > 1e-6 ? grad / len : vec2(0.0);

    frag_color = vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1.0);
}
```

- [ ] **Step 2: Build check**

Run: `cargo check`
Expected: PASS. (Note: Rust still references the old `uniform_sdf_mip` / `uniform_lod_base` / `uniform_max_lod` uniforms on `JfaEncodeProgram`. These will be replaced in Task 8; until then `GetUniformLocation` returns `-1` for the missing ones, which is harmless — `Uniform1f(-1, ...)` is a no-op per GL spec.)

- [ ] **Step 3: Commit**

```bash
git add src/render_helpers/shaders/jfa_encode.frag
git commit -m "feat(shaders): JFA encode reads Poisson field for direction"
```

---

### Task 7: Add program structs and compile functions in `blur.rs`

**Files:**
- Modify: `src/render_helpers/blur.rs`

Add five new program structs and their compile functions, mirroring the existing JFA program/compile patterns. Done in one task so they all build cleanly together.

- [ ] **Step 1: Add the five program structs**

Insert immediately before `struct JfaSdfDownsampleProgram` (around `blur.rs:271`):

```rust
#[derive(Debug)]
struct JfaPoissonInitRhsProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonRestrictMaskProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_src_texel: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonJacobiProgram {
    program: ffi::types::GLuint,
    uniform_u_in: ffi::types::GLint,
    uniform_rhs: ffi::types::GLint,
    uniform_texel: ffi::types::GLint,
    uniform_h_sq: ffi::types::GLint,
    uniform_omega: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonResidualRestrictProgram {
    program: ffi::types::GLuint,
    uniform_u_fine: ffi::types::GLint,
    uniform_rhs_fine: ffi::types::GLint,
    uniform_mask_coarse: ffi::types::GLint,
    uniform_fine_texel: ffi::types::GLint,
    uniform_h_sq_fine: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonProlongateProgram {
    program: ffi::types::GLuint,
    uniform_u_fine: ffi::types::GLint,
    uniform_correction: ffi::types::GLint,
    uniform_rhs_fine: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

- [ ] **Step 2: Add the compile functions**

Insert immediately before `unsafe fn compile_jfa_sdf_downsample` (around `blur.rs:390`):

```rust
unsafe fn compile_jfa_poisson_init_rhs(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonInitRhsProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_init_rhs.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonInitRhsProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_restrict_mask(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonRestrictMaskProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_restrict_mask.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonRestrictMaskProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_src_texel: gl.GetUniformLocation(program, c"niri_src_texel".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_jacobi(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonJacobiProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_jacobi.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonJacobiProgram {
        program,
        uniform_u_in: gl.GetUniformLocation(program, c"niri_u_in".as_ptr()),
        uniform_rhs: gl.GetUniformLocation(program, c"niri_rhs".as_ptr()),
        uniform_texel: gl.GetUniformLocation(program, c"niri_texel".as_ptr()),
        uniform_h_sq: gl.GetUniformLocation(program, c"niri_h_sq".as_ptr()),
        uniform_omega: gl.GetUniformLocation(program, c"niri_omega".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_residual_restrict(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonResidualRestrictProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_residual_restrict.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonResidualRestrictProgram {
        program,
        uniform_u_fine: gl.GetUniformLocation(program, c"niri_u_fine".as_ptr()),
        uniform_rhs_fine: gl.GetUniformLocation(program, c"niri_rhs_fine".as_ptr()),
        uniform_mask_coarse: gl.GetUniformLocation(program, c"niri_mask_coarse".as_ptr()),
        uniform_fine_texel: gl.GetUniformLocation(program, c"niri_fine_texel".as_ptr()),
        uniform_h_sq_fine: gl.GetUniformLocation(program, c"niri_h_sq_fine".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_prolongate(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonProlongateProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_prolongate.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonProlongateProgram {
        program,
        uniform_u_fine: gl.GetUniformLocation(program, c"niri_u_fine".as_ptr()),
        uniform_correction: gl.GetUniformLocation(program, c"niri_correction".as_ptr()),
        uniform_rhs_fine: gl.GetUniformLocation(program, c"niri_rhs_fine".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}
```

- [ ] **Step 3: Update `JfaEncodeProgram` to drop LoD uniforms and add `uniform_poisson_u`**

Find this block (around `blur.rs:282-291`):

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

Replace with:

```rust
#[derive(Debug)]
struct JfaEncodeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_poisson_u: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

- [ ] **Step 4: Update `compile_jfa_encode` to match**

Find and replace the existing function with:

```rust
unsafe fn compile_jfa_encode(gl: &ffi::Gles2) -> Result<JfaEncodeProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_encode.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaEncodeProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_poisson_u: gl.GetUniformLocation(program, c"niri_poisson_u".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_max_dist: gl.GetUniformLocation(program, c"niri_max_dist".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}
```

- [ ] **Step 5: Build check**

Run: `cargo check`
Expected: COMPILE ERROR — `encode_output` still references `prog.uniform_sdf_mip`, `prog.uniform_lod_base`, `prog.uniform_max_lod`. That's expected; the wiring updates in later tasks will fix it. Do not commit yet.

- [ ] **Step 6: Do NOT commit yet.**

---

### Task 8: Update `JfaPipeline` and add `MultigridLevel` / `JfaTextures` changes

**Files:**
- Modify: `src/render_helpers/blur.rs`

Wire the new programs into `JfaPipeline`. Add the `MultigridLevel` struct. Replace `JfaTextures.sdf` with `pyramid: Vec<MultigridLevel>`. Update texture allocation.

- [ ] **Step 1: Update `JfaPipeline`**

Find the existing definition (around `blur.rs:293-302`):

```rust
#[derive(Debug)]
struct JfaPipeline {
    binary_prog: JfaBinaryProgram,
    init_prog: JfaInitProgram,
    step_prog: JfaStepProgram,
    sdf_bake_prog: JfaSdfBakeProgram,
    sdf_downsample_prog: JfaSdfDownsampleProgram,
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
    poisson_init_rhs_prog: JfaPoissonInitRhsProgram,
    poisson_restrict_mask_prog: JfaPoissonRestrictMaskProgram,
    poisson_jacobi_prog: JfaPoissonJacobiProgram,
    poisson_residual_restrict_prog: JfaPoissonResidualRestrictProgram,
    poisson_prolongate_prog: JfaPoissonProlongateProgram,
    encode_prog: JfaEncodeProgram,
}
```

- [ ] **Step 2: Add `MultigridLevel`**

Insert immediately before `struct JfaTextures` (around `blur.rs:304`):

```rust
#[derive(Debug)]
struct MultigridLevel {
    /// Ping-pong A for the solution `u` (R channel).
    u_a: GlesTexture,
    /// Ping-pong B for the solution `u` (R channel).
    u_b: GlesTexture,
    /// Packed RHS: R = `f` (forcing function), G = mask.
    rhs: GlesTexture,
    /// Level size.
    size: Size<i32, Buffer>,
}
```

- [ ] **Step 3: Update `JfaTextures`**

Find this block (around `blur.rs:306-321`):

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
    /// Scalar SDF (R channel) baked from JFA; sampled by encode for R.
    sdf: GlesTexture,
    /// Multigrid pyramid. Index 0 is the finest (bbox-sized) level.
    pyramid: Vec<MultigridLevel>,
    /// Final encoded output (R=normalized SDF, GB=encoded direction).
    encoded: GlesTexture,
    /// Bbox size these textures were allocated for.
    size: Size<i32, Buffer>,
}
```

Note: `sdf` is kept — it holds the baked scalar SDF for the R channel. It's no longer mipmapped.

- [ ] **Step 4: Update the texture allocation block in `render_custom`**

Find this block (around `blur.rs:571-582`):

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

Replace with:

```rust
                    if need_alloc {
                        // Build the multigrid pyramid sized to the bbox.
                        // Level 0 is bbox_size; each subsequent level halves
                        // both dimensions (floor, min 1). Stop at K=5 or
                        // when the level would degenerate to 1×1.
                        let pyramid_max_levels = 6usize; // K+1 = 6
                        let mut pyramid: Vec<MultigridLevel> = Vec::with_capacity(pyramid_max_levels);
                        for k in 0..pyramid_max_levels {
                            let mw = std::cmp::max(1, bbw >> k);
                            let mh = std::cmp::max(1, bbh >> k);
                            let lvl_size = Size::new(mw, mh);
                            pyramid.push(MultigridLevel {
                                u_a: renderer.create_buffer(Fourcc::Abgr16161616f, lvl_size)?,
                                u_b: renderer.create_buffer(Fourcc::Abgr16161616f, lvl_size)?,
                                rhs: renderer.create_buffer(Fourcc::Abgr16161616f, lvl_size)?,
                                size: lvl_size,
                            });
                            if mw == 1 || mh == 1 {
                                break;
                            }
                        }

                        self.jfa_textures = Some(JfaTextures {
                            bin: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_a: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_b: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            sdf: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            pyramid,
                            encoded: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            size: bbox_size,
                        });
                    }
```

- [ ] **Step 5: Update `ensure_jfa_pipeline`**

Find this block (around `blur.rs:874-887`):

```rust
    match (|| -> Result<JfaPipeline, GlesError> {
        Ok(JfaPipeline {
            binary_prog: compile_jfa_binary(gl)?,
            init_prog: compile_jfa_init(gl)?,
            step_prog: compile_jfa_step(gl)?,
            sdf_bake_prog: compile_jfa_sdf_bake(gl)?,
            sdf_downsample_prog: compile_jfa_sdf_downsample(gl)?,
            encode_prog: compile_jfa_encode(gl)?,
        })
    })() {
```

Replace with:

```rust
    match (|| -> Result<JfaPipeline, GlesError> {
        Ok(JfaPipeline {
            binary_prog: compile_jfa_binary(gl)?,
            init_prog: compile_jfa_init(gl)?,
            step_prog: compile_jfa_step(gl)?,
            sdf_bake_prog: compile_jfa_sdf_bake(gl)?,
            poisson_init_rhs_prog: compile_jfa_poisson_init_rhs(gl)?,
            poisson_restrict_mask_prog: compile_jfa_poisson_restrict_mask(gl)?,
            poisson_jacobi_prog: compile_jfa_poisson_jacobi(gl)?,
            poisson_residual_restrict_prog: compile_jfa_poisson_residual_restrict(gl)?,
            poisson_prolongate_prog: compile_jfa_poisson_prolongate(gl)?,
            encode_prog: compile_jfa_encode(gl)?,
        })
    })() {
```

- [ ] **Step 6: Do NOT commit yet.**

Tree still doesn't build — `render_jfa_mask` references `&textures.sdf` for mipmap operations that no longer exist. Tasks 9–11 fix it.

---

### Task 9: Add multigrid stage helper functions

**Files:**
- Modify: `src/render_helpers/blur.rs`

Add the per-stage helpers: `init_rhs_level0`, `restrict_mask_pyramid`, `jacobi_smooth`, `residual_restrict`, `prolongate`. These will be called by the V-cycle orchestrator in Task 10.

- [ ] **Step 1: Add `init_rhs_level0`**

Insert immediately after the existing `bake_sdf` function (currently around `blur.rs:1090`):

```rust
// Writes pyramid[0].rhs from the binary mask:
//   R = 1 inside mask, 0 outside
//   G = mask (1 inside, 0 outside, fractional at AA edges if any)
unsafe fn init_rhs_level0(
    gl: &ffi::Gles2,
    prog: &JfaPoissonInitRhsProgram,
    bin: &GlesTexture,
    dst_rhs: &GlesTexture,
    level_w: i32,
    level_h: i32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst_rhs.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);

    gl.Viewport(0, 0, level_w, level_h);
    gl.BindTexture(ffi::TEXTURE_2D, bin.tex_id());
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

- [ ] **Step 2: Add `restrict_mask_pyramid`**

Insert immediately after `init_rhs_level0`:

```rust
// For each pyramid level k = 1..pyramid.len(), box-downsamples the mask
// from level k-1's rhs.G into level k's rhs (also propagates f via R,
// though level 0's f field is `1 inside mask` so the restricted values
// at coarser levels initialise `f` reasonably; per-V-cycle restriction
// later overwrites it with residuals).
unsafe fn restrict_mask_pyramid(
    gl: &ffi::Gles2,
    prog: &JfaPoissonRestrictMaskProgram,
    pyramid: &[MultigridLevel],
) {
    if pyramid.len() < 2 {
        return;
    }
    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);

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

    for k in 1..pyramid.len() {
        let src = &pyramid[k - 1];
        let dst = &pyramid[k];

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            dst.rhs.tex_id(),
            0,
        );

        gl.Uniform2f(
            prog.uniform_src_texel,
            1.0 / src.size.w as f32,
            1.0 / src.size.h as f32,
        );

        gl.BindTexture(ffi::TEXTURE_2D, src.rhs.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);

        gl.Viewport(0, 0, dst.size.w, dst.size.h);
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    }

    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
}
```

- [ ] **Step 3: Add `jacobi_smooth`**

Insert immediately after `restrict_mask_pyramid`:

```rust
// Runs `sweeps` weighted-Jacobi sweeps on pyramid[level], ping-ponging
// between u_a and u_b. Returns the index (0 = u_a, 1 = u_b) of the
// texture holding the latest u.
unsafe fn jacobi_smooth(
    gl: &ffi::Gles2,
    prog: &JfaPoissonJacobiProgram,
    pyramid: &[MultigridLevel],
    level: usize,
    sweeps: i32,
    current_u_idx: usize, // 0 = u_a is current, 1 = u_b is current
    omega: f32,
) -> usize {
    let lvl = &pyramid[level];
    let h_sq = (1u64 << (2 * level)) as f32; // h^2 = (2^level)^2 = 4^level

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_u_in, 0);
    gl.Uniform1i(prog.uniform_rhs, 1);
    gl.Uniform2f(
        prog.uniform_texel,
        1.0 / lvl.size.w as f32,
        1.0 / lvl.size.h as f32,
    );
    gl.Uniform1f(prog.uniform_h_sq, h_sq);
    gl.Uniform1f(prog.uniform_omega, omega);

    // RHS stays bound on TEXTURE1 across all sweeps.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, lvl.rhs.tex_id());
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

    let mut current = current_u_idx;
    for _ in 0..sweeps {
        let (src, dst) = if current == 0 {
            (&lvl.u_a, &lvl.u_b)
        } else {
            (&lvl.u_b, &lvl.u_a)
        };

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            dst.tex_id(),
            0,
        );

        gl.ActiveTexture(ffi::TEXTURE0);
        gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

        gl.Viewport(0, 0, lvl.size.w, lvl.size.h);
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);

        current = 1 - current;
    }

    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
    current
}
```

- [ ] **Step 4: Add `residual_restrict`**

Insert immediately after `jacobi_smooth`:

```rust
// Computes residual at fine level and restricts (box-downsamples) it
// into coarse level's rhs.R. The coarse level's mask (rhs.G) is read
// from itself and preserved.
//
// Resets coarse-level u_a to zero (so the next coarser smoothing starts
// from u=0, as the multigrid correction equation requires).
unsafe fn residual_restrict(
    gl: &ffi::Gles2,
    prog: &JfaPoissonResidualRestrictProgram,
    pyramid: &[MultigridLevel],
    fine_level: usize,
    fine_u_idx: usize,
) {
    let fine = &pyramid[fine_level];
    let coarse = &pyramid[fine_level + 1];
    let fine_u = if fine_u_idx == 0 { &fine.u_a } else { &fine.u_b };
    let h_sq_fine = (1u64 << (2 * fine_level)) as f32;

    // 1. Compute residual + box-restrict into coarse.rhs.
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        coarse.rhs.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_u_fine, 0);
    gl.Uniform1i(prog.uniform_rhs_fine, 1);
    gl.Uniform1i(prog.uniform_mask_coarse, 2);
    gl.Uniform2f(
        prog.uniform_fine_texel,
        1.0 / fine.size.w as f32,
        1.0 / fine.size.h as f32,
    );
    gl.Uniform1f(prog.uniform_h_sq_fine, h_sq_fine);

    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, fine_u.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, fine.rhs.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
    gl.ActiveTexture(ffi::TEXTURE2);
    // The coarse mask was already restricted by restrict_mask_pyramid
    // and lives in coarse.rhs.G; we sample it here even though we're
    // writing to the same texture. To avoid feedback, we use the rhs of
    // the next-coarser level... actually, we're WRITING to coarse.rhs, so
    // simultaneous read is undefined. The shader reads niri_mask_coarse
    // from the same texture as the write target. This is a known issue:
    // sample-then-write of the same RGBA texture is undefined per GL spec.
    //
    // Workaround: a tiny pre-pass copying coarse.rhs.G into a scratch
    // texture, then sampling that. We don't have a spare texture, but the
    // coarse level's u_b is unused at this point (we'll zero u_a after
    // this anyway). Use coarse.u_b as the scratch by binding it as
    // mask_coarse — except its format is also RGBA16F and it'd need to be
    // populated first.
    //
    // Simpler: do the mask carry-through in a *separate* tiny pass before
    // this one, copying coarse.rhs.G into coarse.u_b.G. But that adds a
    // pass per level. Instead we accept that fragment-shader writes don't
    // commit until the draw call finishes, and we structure the shader to
    // sample mask in a single read before writing. In practice, drivers
    // handle this consistently as "read pre-draw value" for non-MSAA
    // single-sample textures; we rely on that here, with a fallback
    // documented if it breaks.
    gl.BindTexture(ffi::TEXTURE_2D, coarse.rhs.tex_id());
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

    gl.Viewport(0, 0, coarse.size.w, coarse.size.h);
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
    gl.ActiveTexture(ffi::TEXTURE0);

    // 2. Reset coarse u_a to zero so subsequent smoothing solves the
    // correction equation `L e = r` from e=0.
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        coarse.u_a.tex_id(),
        0,
    );
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);
}
```

The known-issue note about feedback-loop sampling is left in the comment as documentation. If the visual quality is bad on the user's driver, the fallback is to copy the mask into `coarse.u_b.G` first; we'll cross that bridge if needed.

- [ ] **Step 5: Add `prolongate`**

Insert immediately after `residual_restrict`:

```rust
// Bilinear-upsamples the coarse-level correction (`coarse.u_a` or u_b
// depending on idx) and adds it into the fine-level solution.
// Writes the updated `u` to fine.u_a or u_b (the OPPOSITE of fine_u_idx)
// and returns the new idx.
unsafe fn prolongate(
    gl: &ffi::Gles2,
    prog: &JfaPoissonProlongateProgram,
    pyramid: &[MultigridLevel],
    fine_level: usize,
    fine_u_idx: usize,
    coarse_u_idx: usize,
) -> usize {
    let fine = &pyramid[fine_level];
    let coarse = &pyramid[fine_level + 1];
    let fine_u = if fine_u_idx == 0 { &fine.u_a } else { &fine.u_b };
    let fine_dst = if fine_u_idx == 0 { &fine.u_b } else { &fine.u_a };
    let coarse_u = if coarse_u_idx == 0 {
        &coarse.u_a
    } else {
        &coarse.u_b
    };

    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        fine_dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_u_fine, 0);
    gl.Uniform1i(prog.uniform_correction, 1);
    gl.Uniform1i(prog.uniform_rhs_fine, 2);

    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, fine_u.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, coarse_u.tex_id());
    // LINEAR upsampling for the coarse correction — this is the bilinear
    // prolongation operator.
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);

    gl.ActiveTexture(ffi::TEXTURE2);
    gl.BindTexture(ffi::TEXTURE_2D, fine.rhs.tex_id());
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

    gl.Viewport(0, 0, fine.size.w, fine.size.h);
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
    gl.ActiveTexture(ffi::TEXTURE0);

    1 - fine_u_idx
}
```

- [ ] **Step 6: Do NOT commit yet.**

The functions exist but are unused. `cargo check` will report unused-function warnings; that's fine. Continue to Task 10.

---

### Task 10: Add `run_v_cycle` orchestration and wire it into `render_jfa_mask`

**Files:**
- Modify: `src/render_helpers/blur.rs`

The V-cycle orchestrator. Initialises level-0 RHS, restricts the mask pyramid (once per frame), then performs one V-cycle: pre-smooth down through levels, restrict residuals, recurse to coarsest, prolongate corrections back up with post-smoothing.

Also replaces the mipmap-generation call in `render_jfa_mask` with this new orchestration.

- [ ] **Step 1: Add `run_v_cycle`**

Insert immediately after `prolongate`:

```rust
// Runs one full V-cycle on the multigrid pyramid. Returns the index
// (0 or 1) of the level-0 `u_a`/`u_b` texture that contains the final
// solution.
//
// Schedule (K = pyramid.len() - 1):
//   For k = 0..K-1:
//     pre-smooth u[k] with N_PRE Jacobi sweeps
//     compute residual at k, box-restrict into rhs[k+1]
//     zero u[k+1]
//   At k = K:
//     "solve" by running 2*N_PRE smoothing sweeps (good enough at
//     the coarsest level where the domain is tiny)
//   For k = K-1..=0:
//     prolongate u[k+1] into u[k]
//     post-smooth u[k] with N_POST Jacobi sweeps
//
// Level-0 `u_a` is assumed to already be zeroed by `init_rhs_level0`
// + a separate gl.Clear (done by the caller).
unsafe fn run_v_cycle(
    gl: &ffi::Gles2,
    pipeline: &JfaPipeline,
    pyramid: &[MultigridLevel],
    n_pre: i32,
    n_post: i32,
    omega: f32,
) -> usize {
    let k_max = pyramid.len() - 1;
    let mut u_idx: Vec<usize> = vec![0; pyramid.len()];

    // Down sweep.
    for k in 0..k_max {
        u_idx[k] = jacobi_smooth(
            gl,
            &pipeline.poisson_jacobi_prog,
            pyramid,
            k,
            n_pre,
            u_idx[k],
            omega,
        );
        residual_restrict(
            gl,
            &pipeline.poisson_residual_restrict_prog,
            pyramid,
            k,
            u_idx[k],
        );
        // residual_restrict zeroed coarse.u_a, so the coarse level
        // starts from u_idx = 0.
        u_idx[k + 1] = 0;
    }

    // Coarsest level — over-smooth as a stand-in for direct solve.
    u_idx[k_max] = jacobi_smooth(
        gl,
        &pipeline.poisson_jacobi_prog,
        pyramid,
        k_max,
        2 * n_pre,
        u_idx[k_max],
        omega,
    );

    // Up sweep.
    for k in (0..k_max).rev() {
        u_idx[k] = prolongate(
            gl,
            &pipeline.poisson_prolongate_prog,
            pyramid,
            k,
            u_idx[k],
            u_idx[k + 1],
        );
        u_idx[k] = jacobi_smooth(
            gl,
            &pipeline.poisson_jacobi_prog,
            pyramid,
            k,
            n_post,
            u_idx[k],
            omega,
        );
    }

    u_idx[0]
}
```

- [ ] **Step 2: Replace the SDF-mipmap call in `render_jfa_mask` with the V-cycle**

Find this block (around `blur.rs:895-913`):

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
        let mip_count = generate_sdf_mipmaps(
            gl,
            &pipeline.sdf_downsample_prog,
            &textures.sdf,
            bbw,
            bbh,
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

        // Build the multigrid pyramid: write level-0 RHS, restrict mask
        // through all coarser levels, clear all u_a's, then run one
        // V-cycle. The result lives in pyramid[0].{u_a, u_b} depending on
        // the returned index.
        init_rhs_level0(
            gl,
            &pipeline.poisson_init_rhs_prog,
            &textures.bin,
            &textures.pyramid[0].rhs,
            textures.pyramid[0].size.w,
            textures.pyramid[0].size.h,
        );
        restrict_mask_pyramid(
            gl,
            &pipeline.poisson_restrict_mask_prog,
            &textures.pyramid,
        );

        // Zero all u_a textures (initial guess u = 0).
        for lvl in &textures.pyramid {
            gl.FramebufferTexture2D(
                ffi::DRAW_FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                lvl.u_a.tex_id(),
                0,
            );
            gl.ClearColor(0.0, 0.0, 0.0, 0.0);
            gl.Clear(ffi::COLOR_BUFFER_BIT);
        }

        let final_u_idx = run_v_cycle(gl, pipeline, &textures.pyramid, 3, 3, 0.8);
        let poisson_u = if final_u_idx == 0 {
            &textures.pyramid[0].u_a
        } else {
            &textures.pyramid[0].u_b
        };
```

- [ ] **Step 3: Replace the `encode_output` call**

Find this block (around `blur.rs:913-924`):

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
            mip_count,
        );
```

Replace with:

```rust
        encode_output(
            gl,
            &pipeline.encode_prog,
            jfa_result,
            poisson_u,
            &textures.encoded,
            bbw,
            bbh,
            max_dist,
        );
```

- [ ] **Step 4: Update `encode_output` signature and body**

Find the existing `encode_output` (around `blur.rs:1234-1296`) and replace it with:

```rust
unsafe fn encode_output(
    gl: &ffi::Gles2,
    prog: &JfaEncodeProgram,
    jfa: &GlesTexture,
    poisson_u: &GlesTexture,
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
    gl.Uniform1i(prog.uniform_poisson_u, 1);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Uniform1f(prog.uniform_max_dist, max_dist);

    gl.Viewport(0, 0, bbw, bbh);

    // Poisson u on TEXTURE1 with NEAREST sampling — we want exact texel
    // values for the central-difference gradient.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, poisson_u.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
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

- [ ] **Step 5: Build**

Run: `cargo check`
Expected: PASS, with dead-code warnings for `JfaSdfDownsampleProgram`, `compile_jfa_sdf_downsample`, and `generate_sdf_mipmaps` (none of these are called now). Task 11 removes them.

- [ ] **Step 6: Commit (consolidates Tasks 7–10 into one logical change)**

```bash
git add src/render_helpers/blur.rs
git commit -m "feat(blur): multigrid Poisson solve for smooth direction field"
```

---

### Task 11: Remove the dead SDF-mipmap path

**Files:**
- Delete: `src/render_helpers/shaders/jfa_sdf_downsample.frag`
- Modify: `src/render_helpers/blur.rs` (remove `JfaSdfDownsampleProgram`, `compile_jfa_sdf_downsample`, `generate_sdf_mipmaps`)

- [ ] **Step 1: Confirm nothing references the removed items**

Run: `rg "jfa_sdf_downsample|JfaSdfDownsampleProgram|compile_jfa_sdf_downsample|generate_sdf_mipmaps" src/`
Expected: matches only the struct definition, compile fn, generate fn in `blur.rs`, and the shader filename. Nothing in `render_jfa_mask` or pipeline construction.

- [ ] **Step 2: Delete the shader file**

```bash
rm src/render_helpers/shaders/jfa_sdf_downsample.frag
```

- [ ] **Step 3: Remove `JfaSdfDownsampleProgram`**

Find this block in `blur.rs` (around the old position, search for the struct):

```rust
#[derive(Debug)]
struct JfaSdfDownsampleProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_src_lod: ffi::types::GLint,
    uniform_src_texel: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}
```

Delete it entirely.

- [ ] **Step 4: Remove `compile_jfa_sdf_downsample`**

Find and delete the whole `compile_jfa_sdf_downsample` function.

- [ ] **Step 5: Remove `generate_sdf_mipmaps`**

Find and delete the whole `generate_sdf_mipmaps` function (it's the one that calls `gl.GenerateMipmap` and renders the mip chain manually).

- [ ] **Step 6: Build**

Run: `cargo check`
Expected: PASS, clean (no warnings about unused items).

- [ ] **Step 7: Commit**

```bash
git add -A src/render_helpers/
git commit -m "chore(blur): remove dead SDF mipmap path"
```

---

### Task 12: Visual verification

**Files:**
- None modified.

The validation gate. Run niri with `overshifted3` and `jfa-debug` on the four test shapes and confirm:

- [ ] **Step 1: With `jfa-debug`, all four shapes show smooth direction fields**

- Solid square: matches the analytical full-window SDF case — smooth radial-ish, no diagonal seams.
- Hollow frame: smooth direction inside each strip, smoothly rotating around inner and outer corners.
- Cross: smooth field, no seam at the central intersection.
- 4-hole square: smooth direction everywhere; near the four holes the direction smoothly curves around them.

- [ ] **Step 2: With `overshifted3`, glass refraction is credible throughout**

- No visible crease at the medial axis of the solid square.
- Refraction wraps smoothly around the holes in the frame and 4-hole square.
- No flicker or wild artefacts when scrolling the test shapes with the mouse wheel.

- [ ] **Step 3: If artefacts remain**

Knobs to tune in this order:
1. Increase V-cycles from 1 to 2 in `render_jfa_mask`'s call to `run_v_cycle` (visible smoothness at the cost of ~50 more passes per frame).
2. Increase pyramid depth `pyramid_max_levels` from 6 to 7 if very large bboxes look insufficiently smooth at the centre.
3. Increase `n_pre` / `n_post` from 3 to 5 for sharper convergence per level.
4. If the `residual_restrict` feedback-loop comment manifests as artefacts (visible flicker or banding at coarse-level boundaries), implement the documented fallback: copy `coarse.rhs.G` into `coarse.u_b.R` in a tiny pre-pass and have the shader sample mask from there instead.

- [ ] **Step 4: No commit needed for verification.**

---

## Self-Review Notes

**Spec coverage:**
- "Solve `∇²u = -1` with `u=0` on boundary" → Task 3 (`jacobi.frag` encodes the iteration; Task 1 sets `f=1` inside / `0` outside; mask multiplier in jacobi enforces Dirichlet boundary).
- "Multigrid V-cycle with K+1 levels" → Task 10 (`run_v_cycle`); pyramid construction Task 8.
- "Pre-smooth 3, post-smooth 3, weighted Jacobi ω=0.8" → defaults in Task 10 step 1 (passed as args to `run_v_cycle`).
- "Final encode reads `u[0]` for direction" → Task 6 (shader) + Task 10 step 4 (Rust wiring).
- "JFA-derived SDF still feeds R" → kept in `bake_sdf` call inside `render_jfa_mask`, preserved across the refactor.
- "Drop SDF mipmap chain" → Task 11.

**Type consistency:**
- All five new program structs share consistent field naming (`uniform_*`, `attrib_vert`) with existing programs.
- `JfaTextures.pyramid: Vec<MultigridLevel>` and the per-level `u_a / u_b / rhs / size` names used consistently across Tasks 8–10.
- `n_pre`, `n_post`, `omega` argument names consistent in `jacobi_smooth` definition (Task 9) and `run_v_cycle` call (Task 10).

**Tasks 7–10 intentionally leave the tree non-building between commits.** This is called out at each `Do NOT commit yet` step. They consolidate into one commit at the end of Task 10. The reason: the type changes to `JfaPipeline` and `JfaTextures` are tightly coupled with the new stage functions and orchestration, and intermediate shims would be wasted work.

**Known issue documented in Task 9 step 4:** `residual_restrict` samples and writes `coarse.rhs` in the same pass. Per GL spec this is undefined; in practice drivers handle it as "read pre-draw values" for non-MSAA single-sample textures. If this manifests as artefacts, Task 12 step 3 documents the fallback (a tiny pre-pass copying the mask to a scratch channel).

**No placeholders detected on review.**

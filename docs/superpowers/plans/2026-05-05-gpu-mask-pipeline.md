# GPU Mask Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move mask generation from CPU EDT at half-res to GPU shader passes at full native resolution, with cheap path for non-overlapping rects and JFA fallback for overlapping unions.

**Architecture:** Two paths selected by overlap detection (O(N²), N≤16, instant). Cheap path: 1 fragment pass computing per-pixel axis-aligned distance + direction. JFA path: 1 seed pass + ~12 ping-pong JFA passes + 1 final combine pass. Both output RGBA8 at full source resolution. Existing down/up/refract/glass pipeline unchanged. Cache persists (skip entire mask pipeline when rects+size unchanged).

**Tech Stack:** Rust, GLES 3.0 fragment shaders, GLSL 300 es.

**Spec:** `docs/superpowers/specs/2026-05-05-gpu-mask-pipeline-design.md`

---

### Task 1: Create mask shader source files

**Files:**
- Create: `src/render_helpers/shaders/mask_vert.vert` — screen-space quad vertex shader
- Create: `src/render_helpers/shaders/mask_cheap.frag` — cheap axis-aligned distance + direction
- Create: `src/render_helpers/shaders/mask_seed.frag` — binary 0/1 seed texture
- Create: `src/render_helpers/shaders/mask_jfa.frag` — one JFA step (nearest-neighbor +8 offsets)
- Create: `src/render_helpers/shaders/mask_final.frag` — sqrt + normalize + direction

### Task 2: Add GPU mask infrastructure to Blur

**Files:**
- Modify: `src/render_helpers/blur.rs` — `Blur` struct, `render_custom`, new methods

Add fields:
- `mask_program_gpu: Option<BlurMaskProgram>` — compiled cheap+JFA shaders
- `jfa_textures: [Option<GlesTexture>; 2]` — JFA ping-pong (RGBA8, pack seed XY)

Add methods:
- `compile_mask_programs(renderer) -> Result<BlurMaskProgram>` — compile all 4 frag shaders with shared vert
- `render_mask_cheap(renderer, options, output_tex)` — 1 pass
- `render_mask_jfa(renderer, options, jfa2)` — seed + N steps + final

Add overlap detection:
- `fn has_overlaps(rects: &[[f32;4]]) -> bool` — O(N²) box intersection test

### Task 3: Update render_custom dispatch

**Files:**
- Modify: `src/render_helpers/blur.rs:494+`

In `render_custom()`:
1. Check cache (rects + size unchanged) → skip mask, just bind cached texture
2. Detect overlaps
3. If no overlaps → `render_mask_cheap`, bind result as niri_mask
4. If overlaps → `render_mask_jfa`, bind result as niri_mask
5. Then run existing down/up/refract/glass passes (unchanged)

Remove all CPU mask code: `generate_mask_data`, `edt_1d`, `cached_mask_data`, mask upload via TexSubImage2D. Mask is now a rendered texture at source resolution.

### Task 4: Remove old generating mask code

**Files:**
- Modify: `src/render_helpers/blur.rs`

Remove: `generate_mask_data` function, `edt_1d` function, `MASK_BLUR_RADIUS` constant, CPU cached mask fields. Update `Blur::new()` to initialize GPU mask fields.

### Task 5: Build and verify

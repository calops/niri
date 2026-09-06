# Custom blur performance notes

The current pipeline architecture is workable, but `region-vectors` spends most of its time after JFA. The full-resolution Poisson solve and full-surface work around it are the main targets.

These estimates come from the render loops and texture formats, not GPU timings. We should add stage-level timings before tuning quality thresholds.

## Current cost

A `region-vectors` cache miss runs:

1. A bbox-local binary mask.
2. JFA initialization.
3. Roughly `log2(max(width, height))` 9-neighbor JFA passes.
4. An SDF bake whose output is not consumed.
5. A six-level Poisson multigrid solve with 12 pre-smoothing and 12 post-smoothing Jacobi sweeps, repeated for two V-cycles.
6. A final encode and blit into a full-source mask.

With the current constants, the Poisson solve submits about 308 draws per cache miss and performs roughly 67 full-bbox-equivalent fragment passes. JFA, setup, encode, and clears bring a 1080p miss to roughly 86 full-bbox-equivalent passes.

The cache avoids JFA and Poisson when bbox-local geometry is unchanged, but every `RegionVectors` invocation still clears the full-source destination and blits the cached bbox. The final mask pipeline isn't cached.

Rigid translation has another avoidable path. The normalized rectangles change, so the outer full-source binary mask is rerendered. The bbox-local JFA cache can still hit, then `RegionVectors` overwrites the binary result without reading it.

## Memory and bandwidth

For a full-screen bbox, the JFA and Poisson textures use approximately 93 bytes per bbox pixel. That is about 193 MB at 1920x1080 and 774 MB at 3840x2160. The two full-source `RGBA16F` mask textures add about 33 MB and 133 MB respectively, before render-pass outputs and Kawase textures.

The biggest waste is the multigrid solution pair. `u_a` and `u_b` are `RGBA32F`, but only R is used. The RHS and binary textures also allocate unused channels.

The JFA coordinates currently use `RGBA16F`. Half-float coordinates stop representing every integer exactly above 2048, so large surfaces can also lose coordinate precision.

## Low-risk changes

### Skip overwritten binary masks

The initial full-source binary pass is only needed when it is the final mask or the first custom mask pass consumes it. `WindowVectors` and `RegionVectors` overwrite their destination and do not read the binary input. If either is the first mask pass, the binary pass can be skipped.

### Cache the final mask

The final mask should be keyed by pipeline identity, source size, rectangles, geometry size, and corner radii. An exact hit can skip the entire binary and mask pipeline.

For translation-only reuse, clear only the previous occupied bbox with a scissor and blit into the new bbox. Clearing the complete source-sized mask defeats much of the bbox-local cache benefit.

A later ABI change could keep the final mask bbox-local and pass a mask UV transform to render shaders, removing the full-source mask blit and potentially the two full-source mask textures.

### Remove the unused SDF bake

`bake_sdf()` writes `JfaTextures::sdf`, but the final encoder reads the JFA result and Poisson solution directly. The program, texture, shader, and draw can be removed.

### Make the JFA step cheaper

`jfa_step.frag` uses `length()` for every candidate even though only distance ordering is needed. Squared distance with `dot(delta, delta)` avoids the square root. Integer coordinates with `texelFetch` also avoid normalized coordinate arithmetic and filtering, and the `(0, 0)` candidate can be skipped because the center was loaded before the loops.

### Retain GL resources and state

Sampler parameters should be set when textures are allocated instead of during each draw. A retained FBO can be reattached to different outputs instead of repeatedly generating and deleting FBOs. The render-pass path currently generates two FBO names while only using one.

Rectangle conversion and upload should happen once when rectangles or source size change, not independently in binary, custom-mask, and JFA paths. `MaskPassStep::Custom::scale` is currently stored and hashed but not applied by the renderer.

## Texture formats

The formats can be narrowed substantially:

- Poisson `u_a` and `u_b`: `R32F` instead of `RGBA32F`.
- Poisson RHS: `RG16F` if precision is sufficient, otherwise `RG32F`.
- Binary mask: `R16F`.
- JFA coordinates: preferably `RG16UI` with an integer sentinel and `texelFetch`, or `RG32F` as the conservative path.
- Final encoded mask: keep `RGBA16F` until the mask contract changes.

Narrowing the Poisson solution alone cuts its dominant solution storage and traffic by 75 percent.

## Poisson improvements

The current pyramid is under-coarsened and over-smoothed. Six levels leave a 1920x1080 solve at about 60x33 and a 4K solve at about 120x67. Those are too large to act as a cheap coarse solve.

We should build levels dynamically until the coarse grid is around 4x4 or 8x8, then try one V-cycle with two to four pre-sweeps and post-sweeps. A modeled 1080p configuration with one cycle and three pre/post sweeps is about 70 draws and 9.7 full-resolution-equivalent passes, compared with 308 draws and about 67 passes now. Visual convergence still needs testing on narrow strips, concave regions, crosses, and disconnected regions.

The safer high-impact option is to retain the full-resolution JFA distance but solve the smooth Poisson direction field at half or quarter linear resolution. A quarter-resolution direction solve reduces Poisson pixel work by about 16 times. The direction can be bilinearly upsampled during the full-resolution encode while distance and coverage remain sharp.

## Rejected Poisson replacement: vector mipmaps

A JFA nearest-boundary vector pyramid was considered as a cheaper direction
field. It would derive vectors from nearest-boundary ownership, build averaged
mip levels, select a distance-dependent LoD, and normalize after sampling.

This approach is rejected for the production region-vector path. Its base field
contains Voronoi discontinuities wherever nearest-boundary ownership changes.
Mip filtering can spread and soften those seams, but cannot guarantee a
continuous field for concave or disconnected regions. Avoiding those seams was
the reason for adopting Poisson. Poisson remains the required direction-field
construction; performance and artifacts must be addressed within its solve,
gradient encoding, and representation.

Switching JFA to an exact EDT could improve the R distance channel but would not
provide the globally smooth direction field supplied by Poisson.

## Binary mask alternative

`mask_binary.frag` loops over every rectangle for every output pixel, so its cost is `bbox pixels * rectangle count`. This is particularly bad for regions represented by many thin rectangles.

We could render instanced analytical rectangle quads with `GL_MAX` blending. Work then follows covered rectangle area and overdraw instead of testing every rectangle at every bbox pixel.

## Process less color data

`FramebufferEffect` captures and processes the full effect geometry. A full-screen transparent layer with a small blur region still incurs a full-screen capture, masks, Kawase blur, and custom render passes.

The long-term fix is to process the union bbox of affected regions plus a sampling halo. Niri can calculate the halo for built-in Kawase passes. Custom render passes need a declared sampling radius or padding, with full-surface processing as the conservative fallback.

For sparse regions on large surfaces, cropping the color pipeline may save more frame time than any JFA micro-optimization.

## Shader-specific notes

### CRT

`crt.frag` samples `niri_input` at `warped` several times for green, alpha, and bloom center. We should fetch it once and reuse the components.

The scanline expression uses `abs(sin(gl_FragCoord.y * pi))`. Fragment centers are normally at `n + 0.5`, so the absolute sine is approximately 1 on every row. The scanline block pays for `sin`, `floor`, `mod`, and a branch while producing effectively no modulation. An explicit integer row period is both correct and cheaper.

The phosphor dots use three `length()` calls and can use squared distances. The sine-based random hash can be replaced with an integer hash or interleaved-gradient noise. The five-sample bloom is also expensive at full resolution and could reuse a blurred/downsampled input.

### Magnify and overshifted glass

Both glass shaders normalize the same 2D light vector more than once. One normalized vector and one signed dot can produce both facing and away terms. Their sine-based noise has the same replacement opportunity as CRT.

`overshifted3/glass.frag` multiplies `to_center` by `center_fade`, then divides by the original magnitude. The result called `dir` is actually `unit_direction * center_fade`, and parts of the lighting are multiplied by `center_fade` again. We should either normalize before applying the fade or keep the magnitude-bearing direction and remove the second fade.

## Mask contract

The final mask ABI is inconsistent:

- The binary mask stores anti-aliased coverage in R.
- `window-vectors` stores normalized interior distance in R and a center vector in GB.
- `region-vectors` stores normalized boundary distance in R and a scaled Poisson direction in GB.
- The example render shaders treat R as a dome/depth coordinate rather than coverage.

A clearer final convention would be:

- R: normalized interior distance.
- GB: unit inward direction encoded to `[0, 1]`.
- A: anti-aliased coverage.

Render shaders can derive center fading from R rather than relying on a hidden magnitude in GB. This also makes a reduced-resolution direction solve much easier to reason about.

## Suggested order

1. Add stage-level GPU timings.
2. Skip overwritten binary masks.
3. Remove the unused SDF stage.
4. Cache the final mask.
5. Optimize JFA candidate comparisons.
6. Narrow texture formats.
7. Test a reduced-resolution Poisson field.
8. Deepen and retune multigrid if full-resolution Poisson remains useful.
9. Test JFA vector mipmaps as the default region field.
10. Crop framebuffer capture and render passes to bbox plus halo.

## Implementation report

### Implemented

- The source-resolution binary pass is skipped when the first mask pass is `window-vectors` or `region-vectors`.
- The completed mask pipeline is cached by compiled pipeline identity, source size, rectangles, geometry size, and corner radii. Exact hits perform no mask GPU work.
- Bbox-local JFA translation hits reuse the encoded field. When the JFA output texture is still its last writer, only the previous bbox is cleared before the new blit. Later ping-pong writers invalidate that ownership marker.
- Pixel-space rectangles are converted and uploaded once per input change, then shared by binary, JFA, and custom mask paths.
- The unused SDF bake program, draw, texture, and shader were removed.
- `jfa_step.frag` now uses integer `texelFetch`, an initialized center sample, eight neighbors, and squared distance comparisons.
- The multigrid pyramid now continues until both coarse dimensions are at most 8 pixels, capped at 12 levels.
- The solver uses one V-cycle with three pre- and three post-Jacobi sweeps. Extra post-sweeps did not remove the visible blocks because they appeared after gradient normalization. The encoder now writes to the unused JFA ping-pong texture, then a cached 3×3 tent pass filters normalized GB direction into the final encoded texture. This adds one bbox-local draw only when the JFA cache misses and does not increase texture precision or memory.
- Poisson solution textures changed from `RGBA32F` to `R32F`, RHS textures from `RGBA16F` to `RG16F`, and bbox binary coverage from `RGBA16F` to `R16F`.
- One framebuffer object is reused across mask and render draws within `render_custom()`. The unused second framebuffer allocation was removed, and raw float textures receive their default sampler state at allocation.
- The example shaders reuse duplicate samples and light-vector calculations. The overshifted glass shader applies center fading once, directly to the lighting direction before constructing the nonlinear specular normal. Magnify uses the same smoothing. CRT scanlines were changed from a no-op expression to a cheap alternating-row response. Noise now uses a nonlinear integer pixel hash; the prior dot/fract hash produced visible diagonal correlation. A weak broad specular lobe was tested as a temporary bridge for highlight gaps, but visual feedback showed no meaningful improvement, so it was removed.

For a 1080p-sized bbox, the Poisson schedule drops from about 308 shader draws to about 71 including the final direction filter. The JFA and Poisson working-set estimate drops from roughly 93 to 42 bytes per bbox pixel, about 193 MB to 87 MB at 1080p and 774 MB to 348 MB at 4K. The encoder fades directions whose bbox-normalized Poisson gradient is between 0.02 and 0.10, then the tent pass removes one-pixel blocks from the normalized vector field. These are static estimates, not GPU measurements.

### Decisions and deferred work

- Vector mipmaps are rejected as a Poisson replacement because smoothing a nearest-boundary field cannot guarantee removal of Voronoi ownership seams.
- Quarter-resolution Poisson remains a fallback if the deeper one-cycle solver is still too expensive or loses too much convergence. Changing both resolution and convergence schedule at once would make visual regressions harder to isolate.
- Cropping the color pipeline to bbox plus halo needs an explicit sampling-radius contract for custom render shaders. Built-in Kawase can calculate its footprint, but arbitrary refraction shaders cannot.
- The final mask ABI still mixes distance, direction, and coverage semantics. Cleaning it up would intentionally break the current shader contract and should be a separate cutover.
- Custom mask-pass `scale` is still parsed and cached but not applied. Supporting it requires scaled mask ping-pong and coordinate transforms rather than ignoring the size mismatch.
- Stage-level GPU timing still needs profiler integration. CPU spans around queued GL work would not produce trustworthy per-stage numbers.

### Verification

- `cargo fmt --all`
- `cargo run --quiet -- validate --config config.kdl`
- `cargo check --all-targets`
- A rebuilt nested niri instance ran an animated changing `region-vectors` layer at 60 updates per second.
- The runtime smoke covered both `region-vectors` plus dual Kawase and `region-vectors` plus dual Kawase plus a custom glass pass. Both pipelines compiled and ran without shader compilation, OpenGL, or panic logs.
- After artifact feedback, a second nested smoke exercised a continuously moving and resizing large arbitrary region through `region-vectors`, dual Kawase, and the custom glass pass. A compositor screenshot was captured for inspection; the pipeline stayed live without shader compilation, OpenGL, or panic logs.
- The Cava fixture queries `niri msg -j outputs`, selects the current mode for its `window.screen.name`, converts niri's millihertz value to hertz, and logs both detected refresh and selected integer FPS. It falls back to 60 FPS if the query fails. Because Winit's Wayland monitor metadata reported the preferred 60 Hz mode on a 170 Hz parent output, the Winit backend accepts `NIRI_WINIT_REFRESH_RATE=<hz>` as an explicit override. `NIRI_CAVA_FRAMERATE=<fps>` independently overrides Cava, allowing compositor timing and region-update frequency to be varied separately. With both overrides set to 170, nested IPC reported 170000 mHz and Cava logged `detected 170 Hz on winit; using 170 FPS`.
- A final nested smoke ran the animated large `cava-glass` region through `region-vectors`, dual Kawase, the custom glass pass, and the cached direction filter. The prior fingerprint-like highlight contours were absent in the captured output, and logs contained no shader compilation, OpenGL, or panic errors.

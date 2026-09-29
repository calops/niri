# Custom shader pipeline performance notes

## Shape routing

Niri owns field construction; configuration does not choose an algorithm.

- A full window uses one analytical rounded-rectangle draw.
- Pairwise-independent simple region rectangles use one instanced analytical draw.
- Touching or overlapping rectangles, connected composite unions, holes, and cursor silhouettes use the bbox-local JFA + Poisson path.

The rectangle classifier sweeps expanded bounds in x order. Expanded bounds include the half-pixel AA footprint, so geometrically touching rectangles cannot take the independent path. While the sweep has found no conflict, active y intervals are disjoint; predecessor and successor checks are sufficient. Classification is $O(n \log n)$.

The analytical rectangle path clears one full field texture, fetches one RGBA32F rect per instance, and rasterizes six vertices per rectangle. No blending is required because the classifier guarantees that expanded instances do not overlap.

## Field ABI

Every producer writes one `RGBA16F` field:

- R: interior distance in logical pixels
- GB: signed inward unit direction multiplied by confidence
- A: authoritative anti-aliased coverage

This removes the old binary/window/region mask contract split. Color shaders can use one implementation for windows, regions, and cursors. Exact output clipping samples A; it never infers coverage from a half-float distance.

The stable field remains full-surface so a partially visible effect can reuse it and change only `niri_field_uv_rect`. Optional `field-shader` stages ping-pong at that stable size.

## JFA + Poisson cost

A composite-region or cursor cache miss runs:

1. bbox-local instanced coverage generation
2. JFA initialization
3. roughly `log2(max(width, height))` eight-neighbor JFA passes
4. one dynamically deep multigrid V-cycle with three pre- and three post-Jacobi sweeps per level
5. field encoding
6. one cached 3×3 GB-only tent filter
7. a bbox blit into the stable full field

The pyramid continues until both coarse dimensions are at most 8 pixels, capped at 12 levels. Poisson solution textures are R32F, RHS textures are RG16F, and bbox coverage is R16F. A 1080p-sized bbox uses roughly 42 bytes of JFA/Poisson working storage per pixel, about 87 MB; 4K is about 348 MB. These are format estimates, not GPU measurements.

The JFA output cache keys bbox size plus bbox-local rectangle offsets, or cursor frame identity and coverage placement. Rigid translation can reuse the solve and only re-blit the encoded field. The completed field cache additionally keys the compiled pipeline, full field size, subregion identity, cursor coverage identity, geometry size, and corner radii.

## Precision constraints

- Multigrid `u` must remain R32F; half-float loses the small gradients used for medial confidence.
- JFA nearest-exterior coordinates remain RGBA16F. Half-float coordinate quantization can create non-zero self-distance outside the shape, so `jfa_encode.frag` uses bbox coverage as authoritative classification.
- Fractional coverage below the 0.5 JFA interior threshold is preserved in A with zero RGB.
- RHS scales by `1/max_dist²` to keep the Poisson solution in a useful numeric range.
- The encoder converts pixel distance to logical pixels before writing R.
- The encoder fades GB with both Poisson-gradient confidence and distance-to-medial-axis confidence. The following tent pass filters GB only.
- Bbox calculations snap with a `1e-4` epsilon to avoid pixel-coordinate wobble.

## Color-stage cost

Color stages still process the full captured effect geometry. A full-screen transparent layer with a small protocol region therefore pays for full-screen capture, blur, and custom color shaders even though field construction is bbox-local.

Cropping color work to a region bbox requires a declared sampling halo. Niri can derive a halo for built-in Kawase blur, but arbitrary custom refraction shaders can sample any source coordinate. Full-surface processing remains the conservative contract.

The `scale=` property applies only to custom color stages. Field shaders deliberately remain at stable full-field resolution so distance units, coverage, cache identity, and `niri_field_uv_rect` do not change between stages.

## Retained optimizations

- Rectangle conversion and RGBA32F upload are shared by analytical and JFA paths.
- JFA candidate comparisons use integer `texelFetch` and squared distance.
- One framebuffer object is reused across field and color draws in `render_custom()`.
- Raw float textures receive retained sampler state where practical.
- The unused standalone SDF bake was removed.
- The exact output pass clips with field A, avoiding thousands of CPU rectangle scissors when post-process noise is zero.
- Custom pipeline compilation failures are cached to prevent per-frame retry storms.

## Remaining work

1. Add GPU timestamp queries for automatic field, JFA, Poisson, custom field, blur, custom color, and output clip stages. CPU spans around queued GL calls are not trustworthy GPU timings.
2. Evaluate half- or quarter-resolution Poisson direction solves while preserving full-resolution JFA distance and coverage.
3. Replace half-float JFA coordinates with an integer representation if large-surface precision becomes visible.
4. Add an explicit custom-shader sampling-radius contract before cropping color stages.

Nearest-boundary vector mipmaps remain rejected as a Poisson replacement. Filtering can soften Voronoi ownership seams but cannot guarantee a continuous direction field for concave or disconnected regions.

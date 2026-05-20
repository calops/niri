#version 300 es

precision highp float;

in vec2 v_coords;

uniform int niri_subregion_count;
// 1D texture holding the subregion rects: one texel per rect, RGBA32F
// = (x1, y1, x2, y2) in source pixels. Sampled with texelFetch so there's
// no fixed compile-time cap on the rect count.
uniform sampler2D niri_subregion_rects;
uniform vec2 niri_output_size;
uniform vec2 niri_bbox_origin;

out vec4 frag_color;

// Analytical anti-aliased rasterisation: for each pixel, compute the
// fraction of its 1×1 area that overlaps with each rect, then take the
// union (approximated as max across rects). Result is in [0, 1]:
//   1.0 — pixel fully inside some rect
//   0.0 — pixel fully outside every rect
//   fractional — pixel straddles a rect boundary
//
// This eliminates the sub-pixel jitter that an integer-threshold mask
// would otherwise produce as shapes shift smoothly under the cursor:
// the Jacobi smoother multiplies u by mask, so a smoothly-varying mask
// produces a smoothly-shifting Poisson boundary.
//
// The max-across-rects approximation is exact for non-overlapping rects.
// For overlapping rects (e.g. the cross's two bars) the true coverage
// can exceed max() (≤ sum, but bounded by 1.0). The approximation
// understates coverage at overlaps by at most ~0.25 of one pixel, which
// is invisible.
void main() {
    vec2 pixel = v_coords * niri_output_size + niri_bbox_origin;
    vec2 pmin = pixel - vec2(0.5);
    vec2 pmax = pixel + vec2(0.5);

    float coverage = 0.0;
    for (int i = 0; i < niri_subregion_count; i++) {
        vec4 r = texelFetch(niri_subregion_rects, ivec2(i, 0), 0);
        vec2 tl = r.xy;
        vec2 br = r.zw;

        vec2 overlap = clamp(min(pmax, br) - max(pmin, tl), 0.0, 1.0);
        float area = overlap.x * overlap.y;
        coverage = max(coverage, area);
    }

    frag_color = vec4(coverage);
}

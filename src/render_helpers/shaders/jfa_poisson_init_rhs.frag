#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;        // binary mask, R = 1 inside, 0 outside
uniform float niri_f_scale;          // RHS amplitude inside the mask

out vec4 frag_color;

void main() {
    float m = texture(niri_input, v_coords).r;
    // Boundary is mask >= 0.5 to keep things sharp at the original
    // resolution; coarser levels may average to fractional values which
    // we'll treat as boundary-weights inside the Jacobi smoother.
    float inside = m >= 0.5 ? 1.0 : 0.0;
    // Scale f so the Poisson solution u stays in a precision-friendly
    // range for RGBA16F. With f = 1 / max_dist^2, u peaks around ~0.06
    // (≈ bbox²/16 ÷ max_dist²) instead of ~bbox²/16 ≈ 10000 at full
    // scale. Quantisation in u that would otherwise wipe out the
    // gradient in deep interior regions is no longer an issue.
    float f = inside * niri_f_scale;
    frag_color = vec4(f, inside, 0.0, 1.0);
}

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

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

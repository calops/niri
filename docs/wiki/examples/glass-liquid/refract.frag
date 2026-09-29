#version 300 es
precision highp float;

in vec2 v_coords;
uniform sampler2D niri_color;

out vec4 frag_color;

void main() {
    vec2 center = v_coords - 0.5;
    float r2 = dot(center, center);
    float r4 = r2 * r2;

    float k1 = -0.1;
    float k2 = 0.05;
    float distortion = 1.0 + k1 * r2 + k2 * r4;

    vec2 distorted = center * distortion + 0.5;
    frag_color = texture(niri_color, distorted);
}

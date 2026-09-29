#version 300 es
precision highp float;

in vec2 v_coords;
uniform sampler2D niri_color;

out vec4 frag_color;

void main() {
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0;
    float refraction_strength = smoothstep(0.0, 1.0, dist) * 0.02;
    vec2 offset = center * refraction_strength;
    vec2 refracted_coords = v_coords + offset;
    frag_color = texture(niri_color, refracted_coords);
}

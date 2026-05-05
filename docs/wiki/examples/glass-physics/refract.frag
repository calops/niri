#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0;

    float refraction_strength = smoothstep(0.0, 1.0, dist) * 0.02;

    vec2 offset = center * refraction_strength;

    vec2 refracted_coords = v_coords + offset;

    gl_FragColor = texture2D(niri_input, refracted_coords);
}

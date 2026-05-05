#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    vec2 center = v_coords - 0.5;
    float r2 = dot(center, center);
    float r4 = r2 * r2;

    float k1 = -0.1;
    float k2 = 0.05;
    float distortion = 1.0 + k1 * r2 + k2 * r4;

    vec2 distorted = center * distortion + 0.5;

    gl_FragColor = texture2D(niri_input, distorted);
}

#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

#define SAMPLE_COUNT 16

void main() {
    float radius = 4.0;
    vec4 color = vec4(0.0);
    float total_weight = 0.0;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float angle = float(i) * 2.39996323;
        float r = radius * sqrt(float(i) / float(SAMPLE_COUNT)) / niri_output_size.x;
        vec2 offset = vec2(cos(angle), sin(angle)) * r;

        float weight = 1.0 - float(i) / float(SAMPLE_COUNT);
        color += texture2D(niri_input, v_coords + offset) * weight;
        total_weight += weight;
    }

    gl_FragColor = color / total_weight;
}

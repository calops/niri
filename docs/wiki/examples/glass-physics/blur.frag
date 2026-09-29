#version 300 es
precision highp float;

in vec2 v_coords;
uniform sampler2D niri_color;
uniform vec2 niri_output_size;

out vec4 frag_color;

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
        color += texture(niri_color, v_coords + offset) * weight;
        total_weight += weight;
    }

    frag_color = color / total_weight;
}

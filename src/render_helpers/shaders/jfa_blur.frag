#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = uv * niri_output_size;

    vec2 st = 1.0 / niri_output_size;

    vec4 sum = vec4(0.0);
    float w = 0.0;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 off = vec2(float(dx), float(dy)) * st;
            vec4 s = texture(niri_input, uv + off);
            if (s.r >= 0.0) {
                sum += s;
                w += 1.0;
            }
        }
    }

    frag_color = w > 0.0 ? sum / w : vec4(-1.0, -1.0, 0.0, 0.0);
}

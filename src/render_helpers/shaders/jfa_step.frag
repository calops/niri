#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform vec2 niri_half_pixel;
uniform int niri_step;

out vec4 frag_color;

void main() {
    vec2 pixel = v_coords * niri_output_size;
    vec2 best = texture(niri_input, v_coords).rg;
    float best_dist = best.x < 0.0 ? 1e10 : length(best - pixel);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 off = vec2(float(dx), float(dy)) * float(niri_step) * niri_half_pixel;
            vec2 cand = texture(niri_input, clamp(v_coords + off, 0.0, 1.0)).rg;
            if (cand.x < 0.0) continue;
            float d = length(cand - pixel);
            if (d < best_dist) {
                best = cand;
                best_dist = d;
            }
        }
    }

    frag_color = vec4(best, 0.0, 1.0);
}

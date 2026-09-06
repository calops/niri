#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform int niri_step;

out vec4 frag_color;

void main() {
    ivec2 pixel_coord = ivec2(gl_FragCoord.xy);
    vec2 pixel = vec2(pixel_coord) + 0.5;
    ivec2 max_coord = ivec2(niri_output_size) - ivec2(1);
    vec2 best = texelFetch(niri_input, pixel_coord, 0).rg;
    vec2 best_delta = best - pixel;
    float best_dist = best.x < 0.0 ? 1e10 : dot(best_delta, best_delta);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            ivec2 sample_coord = clamp(
                pixel_coord + ivec2(dx, dy) * niri_step,
                ivec2(0),
                max_coord
            );
            vec2 cand = texelFetch(niri_input, sample_coord, 0).rg;
            if (cand.x < 0.0) continue;
            vec2 delta = cand - pixel;
            float d = dot(delta, delta);
            if (d < best_dist) {
                best = cand;
                best_dist = d;
            }
        }
    }

    frag_color = vec4(best, 0.0, 1.0);
}

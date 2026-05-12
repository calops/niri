#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform int niri_axis;           // 0 = horizontal, 1 = vertical
uniform int niri_radius_low;     // small radius (e.g. 2)
uniform int niri_radius_high;    // large radius (e.g. 20)

out vec4 frag_color;

// Must be >= max(niri_radius_low, niri_radius_high).
const int MAX_R = 24;

void main() {
    vec2 uv = v_coords;
    vec2 step_uv = (niri_axis == 0)
        ? vec2(1.0 / niri_output_size.x, 0.0)
        : vec2(0.0, 1.0 / niri_output_size.y);

    float sum_low = 0.0;
    float sum_high = 0.0;
    float count_low = 0.0;
    float count_high = 0.0;

    for (int i = -MAX_R; i <= MAX_R; i++) {
        vec2 s_uv = clamp(uv + float(i) * step_uv, 0.0, 1.0);
        vec4 s = texture(niri_input, s_uv);

        // Horizontal pass reads binary mask (R only); both channels accumulate
        // from R. Vertical pass reads the horizontal intermediate: R already
        // low-blurred horizontally, G already high-blurred horizontally.
        float src_low  = (niri_axis == 0) ? s.r : s.r;
        float src_high = (niri_axis == 0) ? s.r : s.g;

        if (abs(i) <= niri_radius_low) {
            sum_low += src_low;
            count_low += 1.0;
        }
        if (abs(i) <= niri_radius_high) {
            sum_high += src_high;
            count_high += 1.0;
        }
    }

    float density_low = count_low > 0.0 ? sum_low / count_low : 0.0;
    float density_high = count_high > 0.0 ? sum_high / count_high : 0.0;

    frag_color = vec4(density_low, density_high, 0.0, 1.0);
}

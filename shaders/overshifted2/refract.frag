#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;
uniform int niri_subregion_count;
uniform vec4 niri_subregion_rects[16];

out vec4 frag_color;

const float M_E = 2.718281828459045;

const float u_a = 0.7;
const float u_b = 2.3;
const float u_c = 5.2;
const float u_d = 6.9;
const float u_fPower = 3.0;

float f(float x) {
    return 1.0 - u_b * pow(u_c * M_E, -u_d * x - u_a);
}

float sdRect(vec2 p) {
    vec2 d = abs(p) - vec2(1.0);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
    vec2 uv = v_coords;

    if (niri_subregion_count == 0) {
        frag_color = texture(niri_input, uv);
        return;
    }

    float min_edge_dist = 1e10;
    vec2 best_center = vec2(0.0);
    vec2 best_half_size = vec2(1.0);

    for (int i = 0; i < niri_subregion_count; i++) {
        vec2 mn = niri_subregion_rects[i].xy;
        vec2 mx = niri_subregion_rects[i].zw;
        vec2 center = (mn + mx) * 0.5;
        vec2 half_size = (mx - mn) * 0.5;
        vec2 d = max(mn - uv, uv - mx);
        float outside = length(max(d, 0.0));
        float inside = min(max(d.x, d.y), 0.0);
        float edge_dist = -(outside + inside);
        if (edge_dist < min_edge_dist) {
            min_edge_dist = edge_dist;
            best_center = center;
            best_half_size = half_size;
        }
    }

    if (min_edge_dist < 0.0) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 p = (uv - best_center) / best_half_size;

    float d = sdRect(p);
    float dist = -d;

    vec2 sampleP = p * pow(f(dist), u_fPower);

    vec2 warped_uv = sampleP * best_half_size + best_center;
    warped_uv = clamp(warped_uv, 0.0, 1.0);

    frag_color = texture(niri_input, warped_uv);
}

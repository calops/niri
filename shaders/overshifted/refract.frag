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

float dist_to_subregions(vec2 uv) {
    float min_dist = 1e10;

    for (int i = 0; i < niri_subregion_count; i++) {
        vec2 mn = niri_subregion_rects[i].xy;
        vec2 mx = niri_subregion_rects[i].zw;
        vec2 d = max(mn - uv, uv - mx);
        float outside_dist = length(max(d, 0.0));
        float inside_dist = min(max(d.x, d.y), 0.0);
        float dist = outside_dist + inside_dist;
        float edge_dist = -dist;
        if (edge_dist < min_dist) {
            min_dist = edge_dist;
        }
    }

    return min_dist;
}

void main() {
    vec2 uv = v_coords;

    if (niri_subregion_count == 0) {
        frag_color = texture(niri_input, uv);
        return;
    }

    float d = dist_to_subregions(uv);

    float eps = 0.0005;
    float d_r = dist_to_subregions(uv + vec2(eps, 0.0));
    float d_u = dist_to_subregions(uv + vec2(0.0, eps));
    vec2 grad = vec2(d_r - d, d_u - d) / eps;
    float grad_len = length(grad);
    vec2 normal = grad_len > 0.001 ? grad / grad_len : vec2(0.0);

    float edge_width = 0.035;
    float max_disp = 0.006;
    float edge_bleed = 0.005;

    float refraction = 0.0;
    if (d > 0.0) {
        refraction = exp(-d / edge_width) * max_disp;
    } else if (d > -edge_bleed) {
        refraction = (1.0 + d / edge_bleed) * max_disp * 0.3;
    }

    vec2 disp = normal * refraction;

    float chromatic = 0.2;
    float r = texture(niri_input, uv - disp * (1.0 - chromatic)).r;
    float g = texture(niri_input, uv - disp).g;
    float b = texture(niri_input, uv - disp * (1.0 + chromatic)).b;
    float a = texture(niri_input, uv - disp).a;

    frag_color = vec4(r, g, b, a);
}

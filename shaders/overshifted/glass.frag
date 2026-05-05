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
    vec4 color = texture(niri_input, uv);

    if (niri_subregion_count == 0) {
        frag_color = color;
        return;
    }

    float d = dist_to_subregions(uv);

    float glow_width = 0.012;
    float glow_intensity = 0.12;

    float glow = exp(-abs(d) / glow_width) * glow_intensity;
    if (d < -0.005) {
        glow *= exp((d + 0.005) / 0.003);
    }

    color.rgb += vec3(glow);

    if (d > 0.0) {
        color.rgb = mix(color.rgb, color.rgb * vec3(0.97, 0.98, 1.02), 0.08);
    }

    frag_color = color;
}

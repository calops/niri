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

const float u_glowWeight = 0.3;
const float u_glowBias = 0.0;
const float u_glowEdge0 = 0.06;
const float u_glowEdge1 = 0.0;
const float u_noise = 0.06;

float sdRect(vec2 p) {
    vec2 d = abs(p) - vec2(1.0);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

float Glow(vec2 uv, vec2 center) {
    vec2 rel = (uv - center) / ((niri_subregion_rects[0].zw - niri_subregion_rects[0].xy) * 0.5);
    return sin(atan(rel.y, rel.x) - 0.5);
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

    float glow = Glow(uv, best_center);
    float mul = glow * u_glowWeight * smoothstep(u_glowEdge0, u_glowEdge1, dist) + 1.0 + u_glowBias;

    vec4 noise = vec4(vec3(rand(gl_FragCoord.xy * 1e-3) - 0.5), 0.0);
    vec4 color = texture(niri_input, uv) + noise * u_noise;

    frag_color = color * vec4(vec3(mul), 1.0);
}

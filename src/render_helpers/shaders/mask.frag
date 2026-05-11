#version 300 es

precision highp float;

in vec2 v_coords;

uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;

out vec4 frag_color;

float sdRoundedRect(vec2 p, vec2 half_size, float r) {
    vec2 q = abs(p) - half_size + r;
    return length(max(q, 0.0)) - r + min(max(q.x, q.y), 0.0);
}

void main() {
    vec2 uv = v_coords;

    float best_dist = 0.0;

    float aspect = niri_geo_size.y / niri_geo_size.x;
    vec2 iso = vec2(1.0, aspect);
    float cr_iso = niri_corner_radius.x / niri_geo_size.x;

    vec2 half_size = vec2(0.5);
    vec2 half_iso = half_size * iso;
    float cr = min(cr_iso, min(half_iso.x, half_iso.y));
    float d = -sdRoundedRect((uv - 0.5) * iso, half_iso, cr);
    if (d > 0.0) {
        best_dist = d;
    }

    float max_half = min(half_iso.x, half_iso.y);
    float mask = max_half > 0.0 ? best_dist / max_half : 0.0;

    if (mask < 0.001) {
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    vec2 to_center = vec2(0.5) - uv;
    frag_color = vec4(
        clamp(mask, 0.0, 1.0),
        to_center.x * 0.5 + 0.5,
        to_center.y * 0.5 + 0.5,
        1.0
    );
}

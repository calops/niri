#version 300 es

// Example field-shader that replaces the cursor field with an analytic disc.
// R is logical-pixel interior distance, GB is signed inward unit direction
// times confidence, and A is authoritative anti-aliased coverage.

precision highp float;

in vec2 v_coords;

uniform vec2 niri_geo_size;

out vec4 frag_color;

void main() {
    vec2 point = v_coords * niri_geo_size;
    vec2 center = niri_geo_size * 0.5;
    vec2 to_center = center - point;
    float center_distance = length(to_center);
    float radius = min(niri_geo_size.x, niri_geo_size.y) * 0.44;
    float signed_distance = radius - center_distance;
    float coverage = smoothstep(-0.5, 0.5, signed_distance);

    if (coverage <= 0.0) {
        frag_color = vec4(0.0);
        return;
    }

    vec2 inward = center_distance > 1e-6
        ? to_center / center_distance
        : vec2(0.0);
    float confidence = smoothstep(0.0, 2.0, center_distance);
    frag_color = vec4(
        max(signed_distance, 0.0),
        inward * confidence,
        coverage
    );
}

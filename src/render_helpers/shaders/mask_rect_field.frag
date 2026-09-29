#version 300 es

precision highp float;

uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

flat in vec4 v_rect;

out vec4 frag_color;

void main() {
    vec2 pixel = gl_FragCoord.xy;
    vec2 footprint_min = pixel - vec2(0.5);
    vec2 footprint_max = pixel + vec2(0.5);
    vec2 overlap = clamp(
        min(footprint_max, v_rect.zw) - max(footprint_min, v_rect.xy),
        0.0,
        1.0
    );
    float coverage = overlap.x * overlap.y;
    if (coverage <= 0.0) {
        frag_color = vec4(0.0);
        return;
    }

    vec2 edge_distance = min(pixel - v_rect.xy, v_rect.zw - pixel);
    float distance_px = max(min(edge_distance.x, edge_distance.y), 0.0);
    float logical_per_pixel = min(
        niri_geo_size.x / niri_output_size.x,
        niri_geo_size.y / niri_output_size.y
    );

    vec2 center = 0.5 * (v_rect.xy + v_rect.zw);
    vec2 half_size = 0.5 * (v_rect.zw - v_rect.xy);
    vec2 normalized_from_center = (pixel - center) / max(half_size, vec2(1e-5));

    // Preserve the old linear normalized center flow. 1/sqrt(2) keeps its
    // confidence at or below one at rectangle corners.
    const float INV_SQRT_2 = 0.70710678118;
    vec2 inward_flow = -normalized_from_center * INV_SQRT_2;

    frag_color = vec4(
        distance_px * logical_per_pixel,
        inward_flow,
        coverage
    );
}

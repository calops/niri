#version 300 es

precision highp float;

in vec2 v_coords;

uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;
// Top-left, top-right, bottom-right, bottom-left in logical pixels.
uniform vec4 niri_corner_radius;

out vec4 frag_color;

float cornerRadius(vec2 p, vec2 half_size) {
    float radius;
    if (p.y >= 0.0) {
        radius = p.x < 0.0 ? niri_corner_radius.x : niri_corner_radius.y;
    } else {
        radius = p.x < 0.0 ? niri_corner_radius.w : niri_corner_radius.z;
    }
    return clamp(radius, 0.0, min(half_size.x, half_size.y));
}

float sdRoundedRect(vec2 p, vec2 half_size, float radius) {
    vec2 q = abs(p) - half_size + radius;
    return length(max(q, 0.0)) - radius + min(max(q.x, q.y), 0.0);
}

vec2 roundedRectFlow(
    vec2 p,
    vec2 half_size,
    float radius,
    float distance
) {
    // The exact rounded-SDF normal is valid until its nearest-edge Voronoi
    // boundaries begin at `distance == radius`. Fade to zero by that point:
    // refraction follows the rounded boundary, the interior remains flat, and
    // no ownership seam can reach the color shader.
    vec2 q = abs(p) - half_size + radius;
    vec2 corner = max(q, 0.0);
    float corner_length = length(corner);
    vec2 outward_local;
    if (corner_length > 1e-5) {
        outward_local = corner / corner_length;
    } else {
        outward_local = q.x > q.y
            ? vec2(1.0, 0.0)
            : vec2(0.0, 1.0);
    }

    vec2 quadrant = vec2(
        p.x < 0.0 ? -1.0 : 1.0,
        p.y < 0.0 ? -1.0 : 1.0
    );
    vec2 inward = -outward_local * quadrant;

    if (radius <= 1e-5) {
        return vec2(0.0);
    }
    float confidence = 1.0 - smoothstep(0.0, radius, distance);
    return inward * confidence;
}

void main() {
    vec2 logical = v_coords * niri_geo_size;
    vec2 half_size = 0.5 * niri_geo_size;
    vec2 p = logical - half_size;
    float radius = cornerRadius(p, half_size);
    float signed_distance = sdRoundedRect(p, half_size, radius);
    float distance = max(-signed_distance, 0.0);

    float logical_per_pixel = min(
        niri_geo_size.x / niri_output_size.x,
        niri_geo_size.y / niri_output_size.y
    );
    float coverage = clamp(-signed_distance / logical_per_pixel + 0.5, 0.0, 1.0);

    vec2 inward_flow = roundedRectFlow(p, half_size, radius, distance);

    frag_color = vec4(distance, inward_flow, coverage);
}

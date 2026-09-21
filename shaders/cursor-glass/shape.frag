#version 300 es

// Example custom cursor mask pass that ignores the xcursor silhouette entirely
// and builds its own analytic disc field. R is the normalized interior distance
// and GB is the vector towards the center, matching the built-in vector masks.
//
// Unlike `cursor-vectors` this needs no JFA or Poisson solve, but the shape is
// whatever this shader decides rather than the actual cursor outline.

precision highp float;

in vec2 v_coords;

uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 center = vec2(0.5);

    float aspect = niri_geo_size.y / niri_geo_size.x;
    vec2 iso = vec2(1.0, aspect);

    // Leave a small margin so the direction and lighting have room to fall off.
    float radius = 0.5 - 0.06;
    float d = radius - length((uv - center) * iso);
    if (d <= 0.0) {
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float mask = clamp(d / radius, 0.0, 1.0);
    vec2 to_center = center - uv;
    frag_color = vec4(
        mask,
        to_center.x * 0.5 + 0.5,
        to_center.y * 0.5 + 0.5,
        1.0
    );
}

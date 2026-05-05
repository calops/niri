#version 300 es

precision highp float;

in vec2 v_coords;

uniform int niri_subregion_count;
uniform vec4 niri_subregion_rects[16];

out vec4 frag_color;

// Smooth minimum: eliminates sharp iso-line seams at corners where
// the nearest edge flips from horizontal to vertical.
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

void main() {
    vec2 uv = v_coords;

    float best_dist = 0.0;
    vec2 best_center = vec2(0.5);
    float max_half = 0.0;

    if (niri_subregion_count == 0) {
        float dx = min(uv.x, 1.0 - uv.x);
        float dy = min(uv.y, 1.0 - uv.y);
        best_dist = smin(dx, dy, 0.005);
        best_center = vec2(0.5);
        max_half = 0.5;
    } else {
        for (int i = 0; i < niri_subregion_count; i++) {
            vec4 r = niri_subregion_rects[i];
            float hw = (r.z - r.x) * 0.5;
            float hh = (r.w - r.y) * 0.5;
            float h = min(hw, hh);
            if (h > max_half) {
                max_half = h;
            }

            float dx = min(uv.x - r.x, r.z - uv.x);
            float dy = min(uv.y - r.y, r.w - uv.y);
            if (dx >= 0.0 && dy >= 0.0) {
                // Smooth min avoids sharp seams where nearest-edge flips at corners.
                float d = smin(dx, dy, 0.005);
                if (d > best_dist) {
                    best_dist = d;
                    best_center = (r.xy + r.zw) * 0.5;
                }
            }
        }
    }

    float mask = max_half > 0.0 ? best_dist / max_half : 0.0;

    if (mask < 0.001) {
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    vec2 to_center = best_center - uv;
    frag_color = vec4(
        clamp(mask, 0.0, 1.0),
        to_center.x * 0.5 + 0.5,
        to_center.y * 0.5 + 0.5,
        1.0
    );
}

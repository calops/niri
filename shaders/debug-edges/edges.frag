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

void main() {
    vec4 color = texture(niri_input, v_coords);

    if (niri_subregion_count == 0) {
        // Blue tint = no subregions, using full-area fallback
        color.rgb = mix(color.rgb, vec3(0.2, 0.4, 1.0), 0.3);

        // Show full-area edge distance
        float d = min(min(v_coords.x, 1.0 - v_coords.x), min(v_coords.y, 1.0 - v_coords.y));
        float border = 1.0 - smoothstep(0.0, 0.05, d);
        color.rgb = mix(color.rgb, vec3(1.0, 1.0, 0.0), border * 0.8);
    } else {
        // Red tint = subregions exist
        color.rgb = mix(color.rgb, vec3(1.0, 0.3, 0.3), 0.15);

        // Draw each subregion rect as a colored outline
        for (int i = 0; i < niri_subregion_count; i++) {
            vec2 mn = niri_subregion_rects[i].xy;
            vec2 mx = niri_subregion_rects[i].zw;

            // Is this pixel inside the rect?
            if (v_coords.x >= mn.x && v_coords.x <= mx.x &&
                v_coords.y >= mn.y && v_coords.y <= mx.y) {
                // Distance to edges of this rect
                float dx = min(v_coords.x - mn.x, mx.x - v_coords.x);
                float dy = min(v_coords.y - mn.y, mx.y - v_coords.y);
                float d = min(dx, dy);
                float border = 1.0 - smoothstep(0.0, 0.01, d);
                color.rgb = mix(color.rgb, vec3(1.0, 1.0, 0.0), border);
            }
        }

        // Encode subregion_count as corner marker
        if (v_coords.x < 0.03 && v_coords.y < 0.03) {
            float b = float(niri_subregion_count) / 16.0;
            color = vec4(float(niri_subregion_count) / 4.0, b, 1.0, 1.0);
        }
    }

    frag_color = color;
}

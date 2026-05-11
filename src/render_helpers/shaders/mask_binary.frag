#version 300 es

precision highp float;

in vec2 v_coords;

uniform int niri_subregion_count;
uniform vec4 niri_subregion_rects[64];
uniform vec2 niri_mask_size;
uniform vec2 niri_bbox_origin;

out vec4 frag_color;

void main() {
    vec2 pixel = v_coords * niri_mask_size + niri_bbox_origin;

    for (int i = 0; i < 64; i++) {
        if (i >= niri_subregion_count) break;
        vec4 r = niri_subregion_rects[i];
        vec2 tl = r.xy;
        vec2 br = r.zw;
        if (pixel.x >= tl.x && pixel.x < br.x && pixel.y >= tl.y && pixel.y < br.y) {
            frag_color = vec4(1.0);
            return;
        }
    }

    frag_color = vec4(0.0);
}

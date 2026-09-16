#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec4 niri_mask_uv_rect;

out vec4 frag_color;

void main() {
    vec2 mask_uv = mix(niri_mask_uv_rect.xy, niri_mask_uv_rect.zw, v_coords);
    vec4 m = texture(niri_mask, mask_uv);
    if (m.r < 0.001) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 dir = (m.gb - 0.5) * 2.0;
    float mag = length(dir);
    dir = mag > 1e-6 ? dir / mag : vec2(0.0);

    frag_color = vec4(dir * 0.5 + 0.5, m.r, 1.0);
}

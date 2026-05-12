#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_src_texel;     // 1.0 / source-level size

out vec4 frag_color;

// 4-tap 2×2 box downsample. Both channels averaged the same way.
void main() {
    vec2 uv = v_coords;
    vec2 o = niri_src_texel * 0.5;
    vec4 a = texture(niri_input, uv + vec2(-o.x, -o.y));
    vec4 b = texture(niri_input, uv + vec2( o.x, -o.y));
    vec4 c = texture(niri_input, uv + vec2(-o.x,  o.y));
    vec4 d = texture(niri_input, uv + vec2( o.x,  o.y));
    frag_color = vec4(0.25 * (a.r + b.r + c.r + d.r),
                      0.25 * (a.g + b.g + c.g + d.g),
                      0.0, 1.0);
}

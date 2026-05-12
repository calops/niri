#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;     // source mip level
uniform float niri_src_lod;       // which level to read
uniform vec2 niri_src_texel;      // 1.0 / source-level size

out vec4 frag_color;

// 4-tap box downsample: average a 2x2 block from the source level. Each tap
// is offset by half a source texel from the destination center so the four
// reads land symmetrically inside the 2x2 source neighborhood.
void main() {
    vec2 uv = v_coords;
    vec2 o = niri_src_texel * 0.5;

    float a = textureLod(niri_input, uv + vec2(-o.x, -o.y), niri_src_lod).r;
    float b = textureLod(niri_input, uv + vec2( o.x, -o.y), niri_src_lod).r;
    float c = textureLod(niri_input, uv + vec2(-o.x,  o.y), niri_src_lod).r;
    float d = textureLod(niri_input, uv + vec2( o.x,  o.y), niri_src_lod).r;

    frag_color = vec4(0.25 * (a + b + c + d), 0.0, 0.0, 1.0);
}

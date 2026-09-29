#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_color;
uniform sampler2D niri_field;
uniform vec4 niri_field_uv_rect;

out vec4 frag_color;

void main() {
    vec2 field_uv = mix(niri_field_uv_rect.xy, niri_field_uv_rect.zw, v_coords);
    float coverage = texture(niri_field, field_uv).a;
    frag_color = texture(niri_color, v_coords) * coverage;
}

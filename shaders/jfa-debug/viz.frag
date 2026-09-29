#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_field;
uniform vec4 niri_field_uv_rect;

out vec4 frag_color;

void main() {
    vec2 field_uv = mix(niri_field_uv_rect.xy, niri_field_uv_rect.zw, v_coords);
    vec4 field = texture(niri_field, field_uv);
    if (field.a <= 0.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float magnitude = length(field.gb);
    vec2 direction = magnitude > 1e-6 ? field.gb / magnitude : vec2(0.0);
    frag_color = vec4(
        direction * 0.5 + 0.5,
        clamp(field.r / 64.0, 0.0, 1.0),
        1.0
    );
}

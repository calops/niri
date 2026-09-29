#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_color;
uniform sampler2D niri_field;
uniform vec4 niri_field_uv_rect;

out vec4 frag_color;

void main() {
    vec2 field_uv = mix(niri_field_uv_rect.xy, niri_field_uv_rect.zw, v_coords);
    vec4 field = texture(niri_field, field_uv);
    vec4 color = texture(niri_color, v_coords);

    if (field.a <= 0.0) {
        color.rgb = mix(color.rgb, vec3(1.0, 0.2, 0.2), 0.25);
    } else {
        float edge = 1.0 - smoothstep(0.0, 3.0, field.r);
        color.rgb = mix(color.rgb, vec3(1.0, 1.0, 0.0), edge);
        color.rgb = mix(color.rgb, vec3(0.2, 0.4, 1.0), 0.12);
    }

    frag_color = color;
}

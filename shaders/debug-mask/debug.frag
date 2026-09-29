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
    vec4 background = texture(niri_color, v_coords);

    if (field.a <= 0.0) {
        frag_color = vec4(background.rgb * 0.3, 1.0);
        return;
    }

    float direction_magnitude = length(field.gb);
    vec3 field_color = vec3(0.5, clamp(field.r / 64.0, 0.0, 1.0), 0.5);
    if (direction_magnitude > 1e-6) {
        float angle = atan(field.b, field.g);
        vec3 direction_color = vec3(
            sin(angle) * 0.5 + 0.5,
            clamp(field.r / 64.0, 0.0, 1.0),
            cos(angle) * 0.5 + 0.5
        );
        field_color = mix(
            field_color,
            direction_color,
            smoothstep(0.0, 0.02, direction_magnitude)
        );
    }

    frag_color = vec4(field_color, 1.0);
}

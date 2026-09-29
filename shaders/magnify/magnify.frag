#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_color;
uniform sampler2D niri_field;
uniform vec4 niri_field_uv_rect;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;

    vec2 field_uv = mix(niri_field_uv_rect.xy, niri_field_uv_rect.zw, uv);
    vec4 field = texture(niri_field, field_uv);

    if (field.a <= 0.0) {
        frag_color = texture(niri_color, uv);
        return;
    }

    float depth = clamp(field.r / 64.0, 0.0, 1.0);
    vec2 inward = field.gb;
    float dome_px = depth / (0.12 + 0.88 * depth) * 64.0;
    vec2 displacement = inward * dome_px / niri_input_size;
    float chromatic = pow(1.0 - depth, 3.0) * 0.05;

    vec2 uv_r = clamp(uv + displacement * (1.0 - chromatic), 0.0, 1.0);
    vec2 uv_g = clamp(uv + displacement, 0.0, 1.0);
    vec2 uv_b = clamp(uv + displacement * (1.0 + chromatic), 0.0, 1.0);

    float cr = texture(niri_color, uv_r).r;
    vec4 green_alpha_sample = texture(niri_color, uv_g);
    float cg = green_alpha_sample.g;
    float cb = texture(niri_color, uv_b).b;
    float ca = green_alpha_sample.a;

    frag_color = vec4(cr, cg, cb, ca);
}

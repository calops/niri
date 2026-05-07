#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;

    vec4 mask_sample = texture(niri_mask, uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 to_center_raw = (mask_sample.gb - 0.5) * 2.0;
    float max_half = mask_sample.a;
    vec2 to_center = to_center_raw * max_half;

    float dome = mask / (0.12 + 0.88 * mask) * 0.45;

    vec2 displacement = +to_center * dome;

    float chromatic = pow(1.0 - mask, 3.0) * 0.05;

    vec2 uv_r = clamp(uv + displacement * (1.0 - chromatic), 0.0, 1.0);
    vec2 uv_g = clamp(uv + displacement, 0.0, 1.0);
    vec2 uv_b = clamp(uv + displacement * (1.0 + chromatic), 0.0, 1.0);

    float cr = texture(niri_input, uv_r).r;
    float cg = texture(niri_input, uv_g).g;
    float cb = texture(niri_input, uv_b).b;
    float ca = texture(niri_input, uv_g).a;

    frag_color = vec4(cr, cg, cb, ca);
}

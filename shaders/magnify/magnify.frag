#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

const float u_magnify_strength = 0.15;
const float u_chromatic = 0.06;

void main() {
    vec2 uv = v_coords;

    vec4 mask_sample = texture(niri_mask, uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;

    float normalized = mask * 2.0 - 1.0;
    float dome = normalized * normalized * normalized * u_magnify_strength;

    vec2 displacement = -to_center * dome;

    vec2 uv_r = clamp(uv + displacement * (1.0 - u_chromatic), 0.0, 1.0);
    vec2 uv_g = clamp(uv + displacement, 0.0, 1.0);
    vec2 uv_b = clamp(uv + displacement * (1.0 + u_chromatic), 0.0, 1.0);

    float cr = texture(niri_input, uv_r).r;
    float cg = texture(niri_input, uv_g).g;
    float cb = texture(niri_input, uv_b).b;
    float ca = texture(niri_input, uv_g).a;

    frag_color = vec4(cr, cg, cb, ca);
}

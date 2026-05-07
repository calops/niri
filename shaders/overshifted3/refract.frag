#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;

out vec4 frag_color;

const float M_E = 2.718281828459045;

const float u_a = 0.7;
const float u_b = 1.5;
const float u_c = 5.2;
const float u_d = 6.9;
const float u_fPower = 3.0;

float f(float x) {
    return 1.0 - u_b * pow(u_c * M_E, -u_d * x - u_a);
}

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

    float base = max(f(mask), 0.0);
    float base_warp = pow(base, u_fPower);
    float chromatic = 0.06;

    float warp_r = pow(base, u_fPower * (1.0 - chromatic));
    float warp_b = pow(base, u_fPower * (1.0 + chromatic));

    vec2 uv_r = uv + to_center * (1.0 - warp_r);
    vec2 uv_g = uv + to_center * (1.0 - base_warp);
    vec2 uv_b = uv + to_center * (1.0 - warp_b);

    uv_r = clamp(uv_r, 0.0, 1.0);
    uv_g = clamp(uv_g, 0.0, 1.0);
    uv_b = clamp(uv_b, 0.0, 1.0);

    float cr = texture(niri_input, uv_r).r;
    float cg = texture(niri_input, uv_g).g;
    float cb = texture(niri_input, uv_b).b;
    float ca = texture(niri_input, uv_g).a;

    frag_color = vec4(cr, cg, cb, ca);
}

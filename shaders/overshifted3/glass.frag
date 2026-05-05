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

const float u_noise = 0.01;

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = v_coords;

    vec4 mask_sample = texture(niri_mask, uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;
    float dist_center = length(to_center);

    float angle = dist_center > 1e-6 ? atan(to_center.y, to_center.x) : 0.0;
    float glow = pow(max(sin(angle - 0.5), 0.0), 3.0) - pow(max(-sin(angle - 0.5), 0.0), 3.0);
    float glow_weight = 0.6;
    float mul = glow * glow_weight * smoothstep(0.2, 0.0, mask) + 1.0;

    vec4 noise = vec4(vec3(rand(gl_FragCoord.xy * 1e-3) - 0.5), 0.0);
    vec4 color = texture(niri_input, uv) + noise * u_noise;

    color.rgb *= mul;

    frag_color = color;
}

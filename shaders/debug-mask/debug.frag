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

    vec4 bg = texture(niri_input, uv);

    if (mask < 0.001) {
        frag_color = vec4(bg.rgb * 0.3, 1.0);
        return;
    }

    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;
    float angle = atan(to_center.y, to_center.x);

    vec3 mask_color = vec3(
        (sin(angle) * 0.5 + 0.5),
        mask,
        (cos(angle) * 0.5 + 0.5)
    );

    frag_color = vec4(mask_color, 1.0);
}

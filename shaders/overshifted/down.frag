#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = 1.0 / niri_input_size;

    vec4 sum = texture(niri_input, uv) * 4.0;
    sum += texture(niri_input, uv + vec2(-pixel.x, -pixel.y));
    sum += texture(niri_input, uv + vec2( pixel.x, -pixel.y));
    sum += texture(niri_input, uv + vec2(-pixel.x,  pixel.y));
    sum += texture(niri_input, uv + vec2( pixel.x,  pixel.y));

    frag_color = sum / 8.0;
}

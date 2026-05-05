#version 100

precision highp float;

varying vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = niri_half_pixel * 2.0;

    vec4 sum = texture2D(niri_input, uv) * 4.0;
    sum += texture2D(niri_input, uv + vec2(-pixel.x, -pixel.y));
    sum += texture2D(niri_input, uv + vec2( pixel.x, -pixel.y));
    sum += texture2D(niri_input, uv + vec2(-pixel.x,  pixel.y));
    sum += texture2D(niri_input, uv + vec2( pixel.x,  pixel.y));

    gl_FragColor = sum / 8.0;
}

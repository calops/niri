#version 100

precision highp float;

varying vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

void main() {
    vec2 uv = v_coords;
    vec2 o = niri_half_pixel * 2.0;

    vec4 sum = vec4(0.0);

    sum += texture2D(niri_input, uv + vec2(-o.x * 2.0, 0.0));
    sum += texture2D(niri_input, uv + vec2( o.x * 2.0, 0.0));
    sum += texture2D(niri_input, uv + vec2(0.0, -o.y * 2.0));
    sum += texture2D(niri_input, uv + vec2(0.0,  o.y * 2.0));

    sum += texture2D(niri_input, uv + vec2(-o.x,  o.y)) * 2.0;
    sum += texture2D(niri_input, uv + vec2( o.x,  o.y)) * 2.0;
    sum += texture2D(niri_input, uv + vec2(-o.x, -o.y)) * 2.0;
    sum += texture2D(niri_input, uv + vec2( o.x, -o.y)) * 2.0;

    gl_FragColor = sum / 12.0;
}

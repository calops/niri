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
    vec2 o = niri_half_pixel * 2.0;

    vec4 sum = vec4(0.0);

    sum += texture(niri_input, uv + vec2(-o.x * 2.0, 0.0));
    sum += texture(niri_input, uv + vec2( o.x * 2.0, 0.0));
    sum += texture(niri_input, uv + vec2(0.0, -o.y * 2.0));
    sum += texture(niri_input, uv + vec2(0.0,  o.y * 2.0));

    sum += texture(niri_input, uv + vec2(-o.x,  o.y)) * 2.0;
    sum += texture(niri_input, uv + vec2( o.x,  o.y)) * 2.0;
    sum += texture(niri_input, uv + vec2(-o.x, -o.y)) * 2.0;
    sum += texture(niri_input, uv + vec2( o.x, -o.y)) * 2.0;

    frag_color = sum / 12.0;
}

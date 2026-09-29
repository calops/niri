#version 300 es
precision highp float;

in vec2 v_coords;
uniform sampler2D niri_color;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

void main() {
    vec2 o = niri_half_pixel * 4.0;
    vec4 sum = texture(niri_color, v_coords) * 4.0;
    sum += texture(niri_color, v_coords + vec2(-o.x, -o.y));
    sum += texture(niri_color, v_coords + vec2( o.x, -o.y));
    sum += texture(niri_color, v_coords + vec2(-o.x,  o.y));
    sum += texture(niri_color, v_coords + vec2( o.x,  o.y));
    frag_color = sum / 8.0;
}

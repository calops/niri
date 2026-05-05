#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_half_pixel;

void main() {
    vec2 o = niri_half_pixel * 4.0;
    vec4 sum = texture2D(niri_input, v_coords) * 4.0;
    sum += texture2D(niri_input, v_coords + vec2(-o.x, -o.y));
    sum += texture2D(niri_input, v_coords + vec2( o.x, -o.y));
    sum += texture2D(niri_input, v_coords + vec2(-o.x,  o.y));
    sum += texture2D(niri_input, v_coords + vec2( o.x,  o.y));
    gl_FragColor = sum / 8.0;
}

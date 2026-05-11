#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;

out vec4 frag_color;

void main() {
    vec4 s = texture(niri_input, v_coords);
    if (s.r > 0.5) {
        vec2 coord = v_coords * niri_output_size;
        frag_color = vec4(coord, 0.0, 1.0);
    } else {
        frag_color = vec4(-1.0, -1.0, 0.0, 0.0);
    }
}

#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform float niri_max_dist;

out vec4 frag_color;

void main() {
    vec2 pixel = v_coords * niri_output_size;
    vec2 nearest = texture(niri_input, v_coords).rg;

    if (nearest.x < 0.0) {
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float dist = length(nearest - pixel);
    float mask = clamp(dist / niri_max_dist, 0.0, 1.0);
    vec2 to_nearest = nearest - pixel;

    frag_color = vec4(
        mask,
        to_nearest.x * 0.5 + 0.5,
        to_nearest.y * 0.5 + 0.5,
        1.0
    );
}

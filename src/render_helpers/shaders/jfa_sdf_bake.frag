#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;   // JFA result
uniform vec2 niri_output_size;
uniform float niri_max_dist;    // normalization scale

out vec4 frag_color;

void main() {
    vec2 pixel = v_coords * niri_output_size;
    vec2 nearest = texture(niri_input, v_coords).rg;

    float d;
    if (nearest.x < 0.0) {
        // Sentinel for "never seen by JFA" — treat as exterior.
        d = 0.0;
    } else {
        d = length(nearest - pixel);
    }

    float sdf = clamp(d / niri_max_dist, 0.0, 1.0);
    frag_color = vec4(sdf, 0.0, 0.0, 1.0);
}

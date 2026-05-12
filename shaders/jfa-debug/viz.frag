#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;

out vec4 frag_color;

void main() {
    vec4 m = texture(niri_mask, v_coords);
    float mask = m.r;

    // Red = exterior (no effect), Green = interior (effect region)
    if (mask < 0.001) {
        frag_color = vec4(0.1, 0.0, 0.0, 1.0); // dark red = outside
    } else if (mask > 0.99) {
        frag_color = vec4(0.0, 0.1, 0.0, 1.0); // dark green = deep inside
    } else {
        frag_color = vec4(mask, mask * 0.5, 0.0, 1.0); // yellow-orange = boundary zone
    }
}

#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;

out vec4 frag_color;

void main() {
    // Built-in region-vectors encodes zero outside and positive distance
    // inside. Convert that field back to the protocol's exact binary clip so
    // the final effect can be drawn as one quad rather than thousands of
    // scissored damage rectangles.
    float coverage = step(0.000001, texture(niri_mask, v_coords).r);
    frag_color = texture(niri_input, v_coords) * coverage;
}

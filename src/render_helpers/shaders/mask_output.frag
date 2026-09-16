#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec4 niri_mask_uv_rect;

out vec4 frag_color;

void main() {
    // Built-in region-vectors encodes zero outside and positive distance
    // inside. Convert that field back to the protocol's exact binary clip so
    // the final effect can be drawn as one quad rather than thousands of
    // scissored damage rectangles.
    vec2 mask_uv = mix(niri_mask_uv_rect.xy, niri_mask_uv_rect.zw, v_coords);
    float coverage = step(0.000001, texture(niri_mask, mask_uv).r);
    frag_color = texture(niri_input, v_coords) * coverage;
}

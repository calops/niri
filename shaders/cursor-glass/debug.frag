#version 300 es

// Debug render pass: paints solid magenta. The automatic cursor field's exact
// GPU clip leaves only the cursor silhouette visible, so a magenta arrow proves
// the effect path, coverage field, and output clip all ran. A plain themed
// cursor means the effect path was not taken (client surface cursor or pipeline
// fallback); no cursor means the field is empty.

precision highp float;

in vec2 v_coords;

out vec4 frag_color;

void main() {
    frag_color = vec4(1.0, 0.0, 1.0, 1.0);
}

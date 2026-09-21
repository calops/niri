#version 300 es

// Debug render pass: paints solid magenta. With a `cursor-vectors` mask pass
// the exact GPU clip leaves only the cursor silhouette visible, so if you see a
// magenta arrow the effect path, the coverage mask, and the output clip all
// work. If you see the plain themed cursor instead, the effect path was not
// taken (client surface cursor or pipeline fallback). If you see nothing, the
// mask is empty.

precision highp float;

in vec2 v_coords;

out vec4 frag_color;

void main() {
    frag_color = vec4(1.0, 0.0, 1.0, 1.0);
}

#version 300 es

precision highp float;

uniform vec2 niri_bbox_origin;

flat in vec4 v_rect;

out vec4 frag_color;

// Each instance covers one rectangle expanded by half a pixel. Compute the
// fraction of this fragment's 1x1 pixel footprint covered by that rectangle.
// GL_MAX blending combines overlapping instances into the same max-union
// coverage produced by the previous per-fragment rectangle loop.
void main() {
    vec2 pixel = gl_FragCoord.xy + niri_bbox_origin;
    vec2 pmin = pixel - vec2(0.5);
    vec2 pmax = pixel + vec2(0.5);
    vec2 overlap = clamp(min(pmax, v_rect.zw) - max(pmin, v_rect.xy), 0.0, 1.0);
    float coverage = overlap.x * overlap.y;
    frag_color = vec4(coverage, 0.0, 0.0, 0.0);
}

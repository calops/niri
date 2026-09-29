#version 300 es

precision highp float;

uniform sampler2D niri_subregion_rects;
uniform vec2 niri_output_size;

flat out vec4 v_rect;

const vec2 QUAD[6] = vec2[6](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0)
);

void main() {
    ivec2 rects_size = textureSize(niri_subregion_rects, 0);
    ivec2 rect_coords = ivec2(gl_InstanceID % rects_size.x, gl_InstanceID / rects_size.x);
    v_rect = texelFetch(niri_subregion_rects, rect_coords, 0);

    // The fragment shader evaluates coverage for every pixel footprint that can
    // intersect the rectangle. The renderer only uses this path when expanded
    // rectangles are pairwise disjoint, so instances need no blending.
    vec2 pixel_min = v_rect.xy - vec2(0.5);
    vec2 pixel_max = v_rect.zw + vec2(0.5);
    vec2 pixel = mix(pixel_min, pixel_max, QUAD[gl_VertexID]);
    gl_Position = vec4(pixel / niri_output_size * 2.0 - 1.0, 1.0, 1.0);
}

#version 300 es

precision highp float;

uniform sampler2D niri_subregion_rects;
uniform vec2 niri_output_size;
uniform vec2 niri_bbox_origin;

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
    v_rect = texelFetch(niri_subregion_rects, ivec2(gl_InstanceID, 0), 0);

    // Cover every destination pixel whose 1x1 footprint can overlap this
    // rectangle. The fragment shader computes the exact overlap fraction.
    vec2 local_min = v_rect.xy - niri_bbox_origin - vec2(0.5);
    vec2 local_max = v_rect.zw - niri_bbox_origin + vec2(0.5);
    vec2 pixel = mix(local_min, local_max, QUAD[gl_VertexID]);
    vec2 position = pixel / niri_output_size * 2.0 - 1.0;
    gl_Position = vec4(position, 1.0, 1.0);
}

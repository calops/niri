#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform int niri_axis;
uniform int niri_radius;

out vec4 frag_color;

// Hard cap on loop bound so the compiler can unroll.
const int MAX_R = 6;

// Separable box blur on all four channels of the encoded SDF +
// direction texture. Smooths out the polygonal corner artefacts that
// the multigrid prolongation introduces from coarse pyramid levels —
// the interior is precision-clean (RGBA32F u) so this blur isn't
// fighting against discrete quantisation any more.
//
// Boundary handling: samples are clamped to [0,1] in UV. The exterior
// sentinel may bleed in by one or two pixels at the rim, but the
// renderer's `mask < 0.001` early-out triggers there anyway.
void main() {
    vec2 uv = v_coords;
    vec2 step_uv = (niri_axis == 0)
        ? vec2(1.0 / niri_output_size.x, 0.0)
        : vec2(0.0, 1.0 / niri_output_size.y);

    vec4 sum = vec4(0.0);
    float count = 0.0;

    for (int i = -MAX_R; i <= MAX_R; i++) {
        if (abs(i) > niri_radius) continue;
        vec2 s_uv = clamp(uv + float(i) * step_uv, 0.0, 1.0);
        sum += texture(niri_input, s_uv);
        count += 1.0;
    }

    frag_color = sum / count;
}

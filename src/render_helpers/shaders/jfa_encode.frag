#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;             // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_sdf_mip;           // mipmapped scalar SDF (R channel)
uniform vec2 niri_output_size;
uniform float niri_max_dist;              // normalization scale for the R channel
uniform float niri_lod_base;              // px of depth per LoD step (e.g. 4.0)
uniform float niri_max_lod;               // clamp ceiling for LoD

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(niri_input, uv).rg;

    if (nearest.x < 0.0) {
        // Exterior sentinel — matches the analytical SDF path.
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float dc = length(nearest - pixel);
    float mask = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Per-pixel LoD: deeper interior → higher mip → smoother field.
    // Logarithmic scaling: lod = log2(max(dc, lod_base) / lod_base)
    // makes the blur radius grow proportionally to depth.
    float lod = log2(max(dc, niri_lod_base) / niri_lod_base);
    lod = clamp(lod, 0.0, niri_max_lod);

    // Central-difference gradient at the chosen LoD. The step must be at
    // least one mip-`lod` texel; otherwise both samples land inside the same
    // mip-`lod` texel and the gradient collapses to per-texel interpolation
    // noise. exp2(lod) is the mip-`lod` texel size in mip-0 texel units.
    vec2 st = exp2(lod) / niri_output_size;
    float r = textureLod(niri_sdf_mip, uv + vec2(st.x, 0.0), lod).r;
    float l = textureLod(niri_sdf_mip, uv - vec2(st.x, 0.0), lod).r;
    float t = textureLod(niri_sdf_mip, uv + vec2(0.0, st.y), lod).r;
    float b = textureLod(niri_sdf_mip, uv - vec2(0.0, st.y), lod).r;

    // Gradient points toward increasing SDF — i.e. inward.
    vec2 grad = vec2(r - l, t - b);
    float len = length(grad);
    vec2 dir = len > 1e-6 ? grad / len : vec2(0.0);

    frag_color = vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1.0);
}

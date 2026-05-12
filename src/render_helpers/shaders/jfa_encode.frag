#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;             // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_density;           // R = density_low, G = density_high
uniform vec2 niri_output_size;
uniform float niri_max_dist;              // normalization scale for the R channel
uniform float niri_edge_threshold_px;     // pixel distance at which blend reaches the high (smooth) gradient

out vec4 frag_color;

float dist_at(vec2 uv) {
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(niri_input, uv).rg;
    if (nearest.x < 0.0) return -1.0;
    return length(nearest - pixel);
}

void main() {
    vec2 uv = v_coords;
    float dc = dist_at(uv);

    if (dc < 0.0) {
        // Exterior sentinel — matches the analytical SDF path so renderers'
        // `mask < 0.001` early-out still triggers.
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float mask = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Central-difference gradient of each density channel, in pixel units.
    vec2 st = 1.0 / niri_output_size;
    vec4 dx = texture(niri_density, uv + vec2(st.x, 0.0))
            - texture(niri_density, uv - vec2(st.x, 0.0));
    vec4 dy = texture(niri_density, uv + vec2(0.0, st.y))
            - texture(niri_density, uv - vec2(0.0, st.y));

    // Density rises toward the interior, so the gradient points inward — exactly
    // what GB encodes for the renderers (`to_center` direction).
    vec2 grad_low = vec2(dx.r, dy.r);
    vec2 grad_high = vec2(dx.g, dy.g);

    float t = smoothstep(0.0, niri_edge_threshold_px, dc);
    vec2 dir = mix(grad_low, grad_high, t);

    float len = length(dir);
    vec2 norm_dir = len > 1e-6 ? dir / len : vec2(0.0);

    frag_color = vec4(mask, norm_dir.x * 0.5 + 0.5, norm_dir.y * 0.5 + 0.5, 1.0);
}

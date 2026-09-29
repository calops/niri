#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;         // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_poisson_u;     // R = Poisson solution u at level 0
uniform sampler2D niri_binary;        // R = authoritative binary coverage
uniform vec2 niri_output_size;
uniform float niri_max_dist;
uniform float niri_logical_per_pixel;
out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(niri_input, uv).rg;
    float coverage = texture(niri_binary, uv).r;

    // Match the classification used by jfa_init.frag. Exterior seeds store
    // their own coordinates in a half-float texture, so recomputing their
    // distance can produce a small positive value rather than exact zero.
    // That numerical residue must not turn the whole JFA bbox into coverage.
    if (coverage <= 0.0 || nearest.x < 0.0) {
        frag_color = vec4(0.0);
        return;
    }
    if (coverage < 0.5) {
        frag_color = vec4(0.0, 0.0, 0.0, coverage);
        return;
    }

    float dc = length(nearest - pixel);
    float normalized_distance = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Direction: central-difference gradient of the Poisson field.
    // ∇u points INTO the interior (toward larger u); that's exactly
    // the "inward" direction the renderers consume. The following pass
    // smooths the normalized encoded vector, where prolongation blocks and
    // half-float direction steps are visible.
    //
    // The Poisson gradient is smooth and continuous away from the medial
    // axis. Its magnitude is a confidence signal: after the RHS is scaled
    // by 1 / max_dist², gmag * max_dist is normally near zero at a medial
    // centre and reaches roughly 0.5 at a boundary. Keep the transition
    // conservative so real gradients are not hidden while tiny coarse-grid
    // residuals cannot become unit vectors.
    //
    // Confidence fades direction near medial axes; distance remains exact.
    vec2 st = 1.0 / niri_output_size;
    float r = texture(niri_poisson_u, uv + vec2(st.x, 0.0)).r;
    float l = texture(niri_poisson_u, uv - vec2(st.x, 0.0)).r;
    float t = texture(niri_poisson_u, uv + vec2(0.0, st.y)).r;
    float b = texture(niri_poisson_u, uv - vec2(0.0, st.y)).r;

    vec2 grad_raw = vec2(r - l, t - b);
    float gmag = length(grad_raw);

    // A normalized gradient below 0.02 is numerical/coarse-grid noise; the
    // fade reaches full confidence at 0.10, well below the expected ~0.5
    // boundary gradient. This smoothstep is deliberately continuous.
    float normalized_gradient = gmag * niri_max_dist;
    float gradient_confidence = smoothstep(0.02, 0.10, normalized_gradient);
    vec2 unit_dir =
        gradient_confidence > 0.0 ? grad_raw / max(gmag, 1e-6) : vec2(0.0);
    float medial_confidence = cos(normalized_distance * 1.57079632679);
    vec2 inward = unit_dir * (medial_confidence * gradient_confidence);

    frag_color = vec4(dc * niri_logical_per_pixel, inward, coverage);
}

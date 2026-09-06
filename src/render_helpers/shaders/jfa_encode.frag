#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;         // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_poisson_u;     // R = Poisson solution u at level 0
uniform vec2 niri_output_size;
uniform float niri_max_dist;          // normalization scale for the R channel

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
    // Magnitude is still derived from the JFA distance field (R channel):
    // it is maximal at boundaries and falls to zero at the centre. The
    // confidence only fades direction; it does not alter the R channel.
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
    float magnitude = cos(mask * 1.57079632679) * 0.2;
    vec2 dir = unit_dir * (magnitude * gradient_confidence);

    frag_color = vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1.0);
}

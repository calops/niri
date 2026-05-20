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
    // the "inward" direction the renderers consume.
    //
    // The Poisson gradient direction is smooth and continuous even
    // through the medial axis of non-convex shapes — this is what the
    // multi-grid solver was built for.  We normalise to a unit vector
    // to isolate direction from convergence-dependent magnitude.
    //
    // Magnitude is derived from the JFA distance field (R channel):
    // it is maximal at boundaries and falls to zero at the centre.
    // This needs no bbox-size calibration or convergence tuning.
    vec2 st = 1.0 / niri_output_size;
    float r = texture(niri_poisson_u, uv + vec2(st.x, 0.0)).r;
    float l = texture(niri_poisson_u, uv - vec2(st.x, 0.0)).r;
    float t = texture(niri_poisson_u, uv + vec2(0.0, st.y)).r;
    float b = texture(niri_poisson_u, uv - vec2(0.0, st.y)).r;

    vec2 grad_raw = vec2(r - l, t - b);
    float gmag = length(grad_raw);
    vec2 unit_dir = gmag > 1e-6 ? grad_raw / gmag : vec2(0.0);
    float magnitude = cos(mask * 1.57079632679) * 0.2;
    vec2 dir = unit_dir * magnitude;

    frag_color = vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1.0);
}

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
    // u peaks somewhere in the interior; ∇u points INTO the interior
    // (toward larger u). That's exactly the "inward" direction the
    // renderers consume via (m.gb - 0.5) * 2.
    //
    // We deliberately DO NOT renormalise. For the Poisson torsion
    // function, |∇u| grows roughly linearly with distance to the nearest
    // boundary, peaking around `max_dist` near edges and falling to zero
    // at interior maxima. Renormalisation would amplify numerical noise
    // wherever the gradient is naturally small (e.g. the centre of a
    // large convex region), producing visible aliasing.
    //
    // Instead, scale the central-difference gradient by 1 / (2 * max_dist)
    // so the result is roughly in [-1, 1]: the `2` accounts for the step
    // being two texels (samples at uv±st), and `max_dist` is the typical
    // gradient magnitude near the boundary. Renderers that want a unit
    // vector can renormalise on their end (glass.frag already does).
    vec2 st = 1.0 / niri_output_size;
    float r = texture(niri_poisson_u, uv + vec2(st.x, 0.0)).r;
    float l = texture(niri_poisson_u, uv - vec2(st.x, 0.0)).r;
    float t = texture(niri_poisson_u, uv + vec2(0.0, st.y)).r;
    float b = texture(niri_poisson_u, uv - vec2(0.0, st.y)).r;

    vec2 grad = vec2(r - l, t - b);
    vec2 dir = grad / (2.0 * niri_max_dist);

    frag_color = vec4(mask, dir.x * 0.5 + 0.5, dir.y * 0.5 + 0.5, 1.0);
}

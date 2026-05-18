#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;         // JFA result (RG = nearest exterior pixel)
uniform sampler2D niri_poisson_u;     // R = Poisson solution u at level 0
uniform vec2 niri_output_size;
uniform float niri_max_dist;

out vec4 frag_color;

// Heatmap: blue(0) → green(0.25) → yellow(0.5) → red(1.0+).
// Pixels below 0.001 (exterior) stay black.
float heat(float v) {
    return clamp(v / 0.5, 0.0, 2.0);
}

void main() {
    vec2 uv = v_coords;
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(niri_input, uv).rg;

    if (nearest.x < 0.0) {
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    vec2 st = 1.0 / niri_output_size;
    float r = texture(niri_poisson_u, uv + vec2(st.x, 0.0)).r;
    float l = texture(niri_poisson_u, uv - vec2(st.x, 0.0)).r;
    float t = texture(niri_poisson_u, uv + vec2(0.0, st.y)).r;
    float b = texture(niri_poisson_u, uv - vec2(0.0, st.y)).r;

    vec2 grad = vec2(r - l, t - b);
    vec2 dir = grad * niri_max_dist * 0.5;
    float grad_mag = length(dir);

    float dc = length(nearest - pixel);
    float mask = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Pack for the jfa-debug viz shader, which decodes as:
    //   dir = normalize((m.gb - 0.5) * 2)
    //   viz_colour = vec4(dir*0.5+0.5, m.r, 1.0)
    //
    // We put the gradient magnitude in R (shown as blue in viz).
    // We put a unit-right direction in GB so the viz shows a
    // constant green background.  Boundary pixels where the
    // analytical |to_center| = 0.5 should read grad_mag ≈ 0.5,
    // which appears as mid-blue in the viz overlay.
    frag_color = vec4(grad_mag, 0.75, 0.5, 1.0);
}

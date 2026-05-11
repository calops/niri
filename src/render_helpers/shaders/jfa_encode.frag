#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_seeds;
uniform vec2 niri_output_size;
uniform float niri_max_dist;

out vec4 frag_color;

float dist_at(sampler2D tex, vec2 uv) {
    vec2 pixel = uv * niri_output_size;
    vec2 nearest = texture(tex, uv).rg;
    if (nearest.x < 0.0) return -1.0;
    return length(nearest - pixel);
}

void main() {
    vec2 uv = v_coords;

    // Distance from raw (unblurred) JFA seeds for accurate mask.
    float dc = dist_at(niri_seeds, uv);
    if (dc < 0.0) {
        frag_color = vec4(0.0, 0.5, 0.5, 1.0);
        return;
    }

    float mask = clamp(dc / niri_max_dist, 0.0, 1.0);

    // Gradient of the *blurred* scalar distance field for smooth direction.
    float s = 2.0;
    vec2 st = s / niri_output_size;

    float dr = dist_at(niri_input, uv + vec2(st.x, 0.0));
    float dl = dist_at(niri_input, uv - vec2(st.x, 0.0));
    float dt = dist_at(niri_input, uv + vec2(0.0, st.y));
    float db = dist_at(niri_input, uv - vec2(0.0, st.y));

    vec2 grad = vec2(dr - dl, dt - db);
    float len = length(grad);

    vec2 to_center = len > 1e-6 ? grad / len : vec2(0.0);

    frag_color = vec4(
        mask,
        to_center.x * 0.5 + 0.5,
        to_center.y * 0.5 + 0.5,
        1.0
    );
}

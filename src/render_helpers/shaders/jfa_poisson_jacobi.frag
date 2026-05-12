#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_u_in;       // R = u at this level
uniform sampler2D niri_rhs;        // R = f, G = mask at this level
uniform vec2 niri_texel;           // 1.0 / level size
uniform float niri_h_sq;           // h^2 at this level (=4^k for level k)
uniform float niri_omega;          // damping (0.8)

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;

    float u_c  = texture(niri_u_in, uv).r;
    float u_l  = texture(niri_u_in, uv - vec2(niri_texel.x, 0.0)).r;
    float u_r  = texture(niri_u_in, uv + vec2(niri_texel.x, 0.0)).r;
    float u_u  = texture(niri_u_in, uv + vec2(0.0, niri_texel.y)).r;
    float u_d  = texture(niri_u_in, uv - vec2(0.0, niri_texel.y)).r;

    vec4 rhs = texture(niri_rhs, uv);
    float f  = rhs.r;
    float m  = rhs.g;

    // ∇²u = -f  ⇒  (u_l + u_r + u_u + u_d - 4 u_c) / h² = -f
    //         ⇒  u_c_new = (u_l + u_r + u_u + u_d + h² f) / 4
    //
    // Weighted Jacobi: u_c_next = (1-ω) u_c + ω u_c_new.
    // Boundary: multiply by `m` so u stays 0 outside the mask.
    float u_jacobi = 0.25 * (u_l + u_r + u_u + u_d + niri_h_sq * f);
    float u_next   = (1.0 - niri_omega) * u_c + niri_omega * u_jacobi;
    frag_color = vec4(u_next * m, 0.0, 0.0, 1.0);
}

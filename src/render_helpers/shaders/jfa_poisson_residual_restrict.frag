#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_u_fine;        // R = u at fine level
uniform sampler2D niri_rhs_fine;      // R = f, G = mask at fine level
uniform vec2 niri_fine_texel;         // 1.0 / fine-level size
uniform float niri_h_sq_fine;         // h^2 at fine level

out vec4 frag_color;

// Sign convention: we solve  ∇²u = -f  →  L u = -f, where L = (sum of 4
// neighbours - 4 * centre) / h². The Jacobi shader expects equations in
// the form `L u = -f`: given `f`, it iterates toward `u` such that
// `L u + f = 0`.
//
// The residual of an approximate u_h is  r = -f - L u_h.
// The coarse correction equation is  L e = r.
// To feed this to the same Jacobi shader, we need to set the coarse-level
// `f` so that `L e = -f_coarse`, i.e.  f_coarse = -r = f_fine + L u_fine.
//
// So we return `f + L u` here (already the negated residual), box-average
// it, and store it directly as the coarse-level RHS. No further sign flip.
float neg_residual_at(vec2 uv) {
    float u_c  = texture(niri_u_fine, uv).r;
    float u_l  = texture(niri_u_fine, uv - vec2(niri_fine_texel.x, 0.0)).r;
    float u_r  = texture(niri_u_fine, uv + vec2(niri_fine_texel.x, 0.0)).r;
    float u_u  = texture(niri_u_fine, uv + vec2(0.0, niri_fine_texel.y)).r;
    float u_d  = texture(niri_u_fine, uv - vec2(0.0, niri_fine_texel.y)).r;
    float f    = texture(niri_rhs_fine, uv).r;
    float lu   = (u_l + u_r + u_u + u_d - 4.0 * u_c) / niri_h_sq_fine;
    return f + lu;
}

void main() {
    vec2 uv = v_coords;
    vec2 o = niri_fine_texel * 0.5;

    float v_a = neg_residual_at(uv + vec2(-o.x, -o.y));
    float v_b = neg_residual_at(uv + vec2( o.x, -o.y));
    float v_c = neg_residual_at(uv + vec2(-o.x,  o.y));
    float v_d = neg_residual_at(uv + vec2( o.x,  o.y));

    // Box-average and store as the coarse `f`. Only the R channel is
    // written (Rust uses glColorMask to preserve G containing the mask
    // populated by restrict_mask_pyramid).
    float neg_r_avg = 0.25 * (v_a + v_b + v_c + v_d);
    frag_color = vec4(neg_r_avg, 0.0, 0.0, 1.0);
}

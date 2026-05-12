#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_u_fine;        // R = u at fine level
uniform sampler2D niri_rhs_fine;      // R = f, G = mask at fine level
uniform sampler2D niri_mask_coarse;   // G = mask at coarse level (pre-restricted)
uniform vec2 niri_fine_texel;         // 1.0 / fine-level size
uniform float niri_h_sq_fine;         // h^2 at fine level

out vec4 frag_color;

// Sign convention: we solve  ∇²u = -f  →  L u = -f, where L = (sum of 4
// neighbours - 4 * centre) / h². The residual of an approximate u is
//   r = -f - L u
// We restrict r to the coarse level so the coarse correction equation
// becomes  L e = r  (same operator form). At the coarse level we then
// start from e = 0 and smooth toward the solution.
float residual_at(vec2 uv) {
    float u_c  = texture(niri_u_fine, uv).r;
    float u_l  = texture(niri_u_fine, uv - vec2(niri_fine_texel.x, 0.0)).r;
    float u_r  = texture(niri_u_fine, uv + vec2(niri_fine_texel.x, 0.0)).r;
    float u_u  = texture(niri_u_fine, uv + vec2(0.0, niri_fine_texel.y)).r;
    float u_d  = texture(niri_u_fine, uv - vec2(0.0, niri_fine_texel.y)).r;
    float f    = texture(niri_rhs_fine, uv).r;
    float lu   = (u_l + u_r + u_u + u_d - 4.0 * u_c) / niri_h_sq_fine;
    return -f - lu;
}

void main() {
    vec2 uv = v_coords;
    vec2 o = niri_fine_texel * 0.5;

    float r_a = residual_at(uv + vec2(-o.x, -o.y));
    float r_b = residual_at(uv + vec2( o.x, -o.y));
    float r_c = residual_at(uv + vec2(-o.x,  o.y));
    float r_d = residual_at(uv + vec2( o.x,  o.y));

    // Box-average to coarse level.
    float r_avg = 0.25 * (r_a + r_b + r_c + r_d);

    // Carry the pre-restricted coarse mask through. (We sample from the
    // same texture we're writing to; per GL spec this is undefined, but
    // in practice drivers return pre-draw values for non-MSAA single-
    // sample textures. Documented in the Rust side; fallback exists if
    // this manifests as artefacts.)
    float m = texture(niri_mask_coarse, uv).g;

    frag_color = vec4(r_avg, m, 0.0, 1.0);
}

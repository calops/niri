#version 300 es

precision highp float;
precision highp int;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;
uniform vec4 niri_window_screen_rect;

out vec4 frag_color;

const float M_E = 2.718281828459045;

const float u_a = 0.7;
const float u_b = 1.5;
const float u_c = 5.2;
const float u_d = 6.9;
const float u_fPower = 3.0;
const float u_chromatic = 0.06;
const float u_noise = 0.005;

// Fade lighting near the Poisson medial axis while retaining the full
// direction signal for refraction. This prevents sharp lighting reversals
// across the centerline without weakening the refraction response.
// Tune up to fade a wider band around the medial axis.
const float u_centerThreshold = 0.1;

const vec2 light_pos = vec2(0.0, 0.7);

float f(float x) {
    return 1.0 - u_b * pow(u_c * M_E, -u_d * x - u_a);
}

uint hash(uvec2 pixel) {
    uint h = pixel.x * 0x8da6b343u ^ pixel.y * 0xd8163841u;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

float rand(uvec2 pixel) {
    return float(hash(pixel)) / 4294967295.0;
}

void main() {
    vec2 uv = v_coords;

    vec4 mask_sample = texture(niri_mask, uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;

    // Fade the lighting direction before constructing the nonlinear
    // specular normal so medial-axis noise cannot become a full-strength
    // highlight. Refraction still uses the full encoded vector.
    float to_center_mag = length(to_center);
    float center_fade = smoothstep(0.0, u_centerThreshold, to_center_mag);
    vec2 unit_dir = to_center_mag > 1e-6 ? to_center / to_center_mag : vec2(0.0);
    vec2 dir = unit_dir * center_fade;

    // --- Chromatic refraction ---
    float base = max(f(mask), 0.0);
    float base_warp = pow(base, u_fPower);
    float warp_r = pow(base, u_fPower * (1.0 - u_chromatic));
    float warp_b = pow(base, u_fPower * (1.0 + u_chromatic));

    vec2 uv_r = clamp(uv + to_center * (1.0 - warp_r), 0.0, 1.0);
    vec2 uv_g = clamp(uv + to_center * (1.0 - base_warp), 0.0, 1.0);
    vec2 uv_b = clamp(uv + to_center * (1.0 - warp_b), 0.0, 1.0);

    float cr = texture(niri_input, uv_r).r;
    vec4 green_alpha_sample = texture(niri_input, uv_g);
    float cg = green_alpha_sample.g;
    float cb = texture(niri_input, uv_b).b;
    float ca = green_alpha_sample.a;

    // --- Lighting ---

    // Surface normal from dome curvature.
    float slope = (1.0 - mask) * 5.0;
    vec3 normal = normalize(vec3(-slope * dir, 3.0));

    // Point light grazing from the configured light position.
    vec2 screen_uv = niri_window_screen_rect.xy + uv * niri_window_screen_rect.zw;
    vec2 to_light = light_pos - screen_uv;
    vec2 light_dir_2d = to_light * inversesqrt(max(dot(to_light, to_light), 1e-12));
    vec3 light_dir = normalize(vec3(to_light, 0.04));
    vec3 half_dir = normalize(light_dir + vec3(0.0, 0.0, 1.0));

    // Dome specular: where the smooth surface faces the half-vector.
    float spec = pow(max(dot(normal, half_dir), 0.0), 60.0);

    // Directional edge glow + shadow: rim brightness depends on how the
    // outward direction aligns with the light. Opposite side gets a
    // shadow.
    float light_alignment = dot(-dir, light_dir_2d);
    float facing = max(light_alignment, 0.0);
    float away = max(-light_alignment, 0.0);
    float edge = pow(1.0 - mask, 3.0);
    float rim_light = facing * edge * 1.5;
    float rim_shadow = away * edge * 0.35;

    float glow = spec * 0.85 + rim_light;

    // --- Compose ---
    vec4 noise = vec4(vec3(rand(uvec2(gl_FragCoord.xy)) - 0.5), 0.0);
    vec4 color = vec4(cr, cg, cb, ca) + noise * u_noise;
    color.rgb *= 1.0 + glow - rim_shadow;

    frag_color = color;
}

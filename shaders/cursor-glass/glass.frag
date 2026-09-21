#version 300 es

// Cursor glass lens. Same model as shaders/overshifted3/glass.frag, applied to
// the cursor silhouette: `niri_mask` is the cursor's interior distance field
// (0 at the outline, 1 at the center) baked by the `cursor-vectors` mask pass,
// and `niri_input` is the padded backdrop captured beneath the cursor so the
// refraction can sample past the outline.

precision highp float;
precision highp int;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec4 niri_mask_uv_rect;
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
const float u_refraction_strength = 1.55;
const float u_chromatic = 0.10;
const float u_edge_chromatic = 0.34;
const float u_noise = 0.005;
const float u_dark_tint = 0.10;
const float u_saturation = 1.45;
const float u_frost = 0.27;
const vec3 u_tint_color = vec3(0.035, 0.045, 0.065);
const vec3 u_frost_tint = vec3(0.68, 0.80, 0.96);

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

vec3 saturate_color(vec3 color, float amount) {
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luminance), color, amount);
}

void main() {
    vec2 uv = v_coords;

    vec2 mask_uv = mix(niri_mask_uv_rect.xy, niri_mask_uv_rect.zw, uv);
    vec4 mask_sample = texture(niri_mask, mask_uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        // Transparent outside the silhouette. This lets the pipeline work both
        // with the exact GPU output clip (mask_output.frag) and with the CPU
        // damage fallback used by custom mask passes.
        frag_color = vec4(0.0);
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
    float edge_chromatic = mix(u_chromatic, u_edge_chromatic, pow(1.0 - mask, 1.6));
    float warp_r = pow(base, u_fPower * (1.0 - edge_chromatic));
    float warp_b = pow(base, u_fPower * (1.0 + edge_chromatic));

    vec2 uv_r = clamp(
        uv + to_center * (1.0 - warp_r) * u_refraction_strength,
        0.0,
        1.0
    );
    vec2 uv_g = clamp(
        uv + to_center * (1.0 - base_warp) * u_refraction_strength,
        0.0,
        1.0
    );
    vec2 uv_b = clamp(
        uv + to_center * (1.0 - warp_b) * u_refraction_strength,
        0.0,
        1.0
    );

    float cr = texture(niri_input, uv_r).r;
    vec4 green_alpha_sample = texture(niri_input, uv_g);
    float cg = green_alpha_sample.g;
    float cb = texture(niri_input, uv_b).b;
    float ca = green_alpha_sample.a;

    // A small tent filter softens the refracted image without obscuring it.
    // The preceding Kawase pass supplies the broader, low-frequency blur.
    vec2 frost_step = niri_half_pixel * 2.5;
    vec4 frost = green_alpha_sample * 0.4;
    frost += texture(niri_input, clamp(uv_g + vec2(frost_step.x, 0.0), 0.0, 1.0)) * 0.15;
    frost += texture(niri_input, clamp(uv_g - vec2(frost_step.x, 0.0), 0.0, 1.0)) * 0.15;
    frost += texture(niri_input, clamp(uv_g + vec2(0.0, frost_step.y), 0.0, 1.0)) * 0.15;
    frost += texture(niri_input, clamp(uv_g - vec2(0.0, frost_step.y), 0.0, 1.0)) * 0.15;
    vec3 refracted_color = mix(vec3(cr, cg, cb), frost.rgb, u_frost);
    float refracted_alpha = mix(ca, frost.a, u_frost);

    // --- Lighting ---

    // Surface normal from dome curvature.
    float slope = (1.0 - mask) * 5.0;
    vec3 normal = normalize(vec3(-slope * dir, 3.0));

    // Point light grazing from the configured light position.
    vec2 screen_uv = niri_window_screen_rect.xy + mask_uv * niri_window_screen_rect.zw;
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

    // Let the silhouette be defined by a directional reflection rather than
    // a uniform stroke. Even the fully lit edge retains its refracted color.
    float edge_band = 1.0 - smoothstep(0.015, 0.075, mask);
    float edge_reflection = edge_band * (0.08 + 0.92 * facing);
    float edge_shadow = edge_band * away * 0.16;
    float glow = spec * 0.85 + rim_light;

    // --- Compose ---
    vec4 noise = vec4(vec3(rand(uvec2(gl_FragCoord.xy)) - 0.5), 0.0);
    vec4 color = vec4(refracted_color, refracted_alpha) + noise * u_noise;
    color.rgb *= 1.0 + glow - rim_shadow;
    color.rgb = saturate_color(color.rgb, u_saturation);
    color.rgb = mix(color.rgb, u_tint_color * color.a, u_dark_tint);
    color.rgb = mix(color.rgb, u_frost_tint * color.a, u_frost * 0.32);
    vec3 edge_reflection_color = saturate_color(
        mix(color.rgb, vec3(color.a), 0.32),
        1.12
    );
    color.rgb = mix(color.rgb, edge_reflection_color, edge_reflection);
    color.rgb *= 1.0 - edge_shadow;

    frag_color = color;
}

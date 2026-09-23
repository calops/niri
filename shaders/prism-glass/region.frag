#version 300 es

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

const float refraction_px = 24.0;
const float chromatic_px = 1.15;
const float frost = 0.34;
const float curve_softness = 0.30;
const float light_radius = 0.52;
const vec3 absorption = vec3(0.075, 0.076, 0.080);
const vec3 body_tint = vec3(0.95, 0.96, 0.98);
const vec3 face_tint = vec3(0.93, 0.95, 0.98);
const vec3 reflection_top = vec3(0.90, 0.93, 0.98);
const vec3 reflection_bottom = vec3(0.32, 0.33, 0.36);
const vec2 light_pos = vec2(0.12, 0.67);

vec4 sampleFrosted(vec2 uv, float radius) {
    vec2 step_uv = vec2(radius) / niri_input_size;
    vec4 color = texture(niri_input, uv) * 0.4;
    color += texture(niri_input, clamp(uv + vec2(step_uv.x, 0.0), 0.0, 1.0)) * 0.15;
    color += texture(niri_input, clamp(uv - vec2(step_uv.x, 0.0), 0.0, 1.0)) * 0.15;
    color += texture(niri_input, clamp(uv + vec2(0.0, step_uv.y), 0.0, 1.0)) * 0.15;
    color += texture(niri_input, clamp(uv - vec2(0.0, step_uv.y), 0.0, 1.0)) * 0.15;
    return color;
}

void main() {
    vec2 uv = v_coords;
    vec2 mask_uv = mix(niri_mask_uv_rect.xy, niri_mask_uv_rect.zw, uv);
    vec4 mask_sample = texture(niri_mask, mask_uv);
    float mask = mask_sample.r;

    if (mask < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    // Preserve the raw smoothed Poisson field, including its deliberate fade
    // near medial axes. Normalizing it recreates exactly the seams the encoder
    // works to suppress.
    vec2 inward = (mask_sample.gb - 0.5) * 2.0;
    vec2 outward = -inward;

    float bevel_position = clamp(mask / 0.24, 0.0, 1.0);
    float circle_x = 1.0 - bevel_position;
    float circle_slope = circle_x
        / sqrt(max(curve_softness + 1.0 - circle_x * circle_x, 1e-4));
    float profile = 1.0 - smoothstep(0.0, 1.0, bevel_position);
    // Keep apparent thickness independent from refraction strength so lighting
    // can describe a deep bevel without folding the sampled backdrop.
    float slope = 3.2 * circle_slope;
    vec3 normal = normalize(vec3(outward * slope, 1.0));

    vec2 offset_uv = inward * (2.0 * refraction_px * profile) / niri_input_size;
    vec2 chroma_uv = inward * (2.0 * chromatic_px * profile) / niri_input_size;
    vec2 refracted_uv = clamp(uv + offset_uv, 0.0, 1.0);

    float red = texture(niri_input, clamp(refracted_uv - chroma_uv, 0.0, 1.0)).r;
    vec4 center = texture(niri_input, refracted_uv);
    float blue = texture(niri_input, clamp(refracted_uv + chroma_uv, 0.0, 1.0)).b;
    vec4 softened = sampleFrosted(refracted_uv, 5.0 + 5.0 * profile);
    vec4 color = mix(vec4(red, center.g, blue, center.a), softened, frost + 0.10 * profile);
    color.rgb = mix(color.rgb, color.rgb * face_tint, 0.045);

    float path = 1.0 / max(normal.z, 0.18) - 1.0;
    vec3 transmission = exp(-absorption * path);
    color.rgb *= transmission;
    float body_density = (1.0 - dot(transmission, vec3(0.2126, 0.7152, 0.0722)))
        * pow(profile, 0.65);
    color.rgb = mix(color.rgb, color.rgb * body_tint, clamp(body_density * 0.42, 0.0, 0.14));

    vec2 screen_uv = niri_window_screen_rect.xy + mask_uv * niri_window_screen_rect.zw;
    vec2 to_light_2d = light_pos - screen_uv;
    float light_distance = length(to_light_2d);
    float area_light = exp(-pow(light_distance / light_radius, 2.0));
    vec3 light_dir = normalize(vec3(to_light_2d, 0.42));
    vec3 half_dir = normalize(light_dir + vec3(0.0, 0.0, 1.0));
    float specular = pow(max(dot(normal, half_dir), 0.0), 10.0) * area_light;
    float fresnel = 0.04 + 0.96 * pow(1.0 - normal.z, 5.0);
    vec3 environment = mix(reflection_top, reflection_bottom, clamp(0.5 + 0.5 * normal.y, 0.0, 1.0));
    environment *= color.a;
    color.rgb = mix(color.rgb, environment, fresnel * 0.72);
    color.rgb += environment * specular * 0.55;

    vec2 light_direction_2d = light_distance > 1e-5
        ? to_light_2d / light_distance
        : vec2(0.0);
    float light_facing = clamp(dot(outward * 2.0, light_direction_2d), 0.0, 1.0);
    float curved_face = pow(profile, 0.65);
    float broad_reflection = curved_face * (0.30 + 0.70 * area_light);
    float shoulder = exp(-pow((bevel_position - 0.58) / 0.26, 2.0));
    float back_surface = exp(-pow((bevel_position - 0.72) / 0.19, 2.0));
    float front_surface = 1.0 - smoothstep(0.0, 0.14, bevel_position);
    float outer_glint = 1.0 - smoothstep(0.006, 0.022, mask);
    vec3 reflection_tint = vec3(0.72, 0.82, 1.0) * color.a;
    color.rgb += reflection_tint * broad_reflection
        * (0.025 + 0.16 * light_facing);
    color.rgb += reflection_tint * shoulder
        * (0.012 + 0.055 * light_facing) * (0.35 + 0.65 * area_light);
    color.rgb += mix(reflection_tint, vec3(0.52, 0.54, 0.58) * color.a, 0.42)
        * back_surface * (0.018 + 0.065 * (1.0 - light_facing));
    color.rgb += vec3(0.88, 0.93, 1.0) * color.a
        * max(front_surface * 0.12, outer_glint * 0.24)
        * (0.25 + 0.75 * light_facing) * (0.45 + 0.55 * area_light);
    color.rgb *= 1.0 - front_surface * (1.0 - light_facing) * 0.10;

    frag_color = color;
}

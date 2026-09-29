#version 300 es

precision highp float;
precision highp int;

in vec2 v_coords;

uniform sampler2D niri_color;
uniform sampler2D niri_field;
uniform vec4 niri_field_uv_rect;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;
uniform vec4 niri_window_screen_rect;

out vec4 frag_color;

const float u_noise = 0.005;
const float u_centerThreshold = 0.1;

const vec2 light_pos = vec2(0.0, 0.7);

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

    vec2 field_uv = mix(niri_field_uv_rect.xy, niri_field_uv_rect.zw, uv);
    vec4 field = texture(niri_field, field_uv);

    if (field.a <= 0.0) {
        frag_color = texture(niri_color, uv);
        return;
    }

    float depth = clamp(field.r / 64.0, 0.0, 1.0);
    vec2 dir = field.gb;

    // --- Lighting ---

    float slope = (1.0 - depth) * 5.0;
    vec3 normal = normalize(vec3(-slope * dir, 3.0));

    vec2 screen_uv = niri_window_screen_rect.xy + field_uv * niri_window_screen_rect.zw;
    vec2 to_light = light_pos - screen_uv;
    vec2 light_dir_2d = to_light * inversesqrt(max(dot(to_light, to_light), 1e-12));
    vec3 light_dir = normalize(vec3(to_light, 0.04));
    vec3 half_dir = normalize(light_dir + vec3(0.0, 0.0, 1.0));

    float spec = pow(max(dot(normal, half_dir), 0.0), 60.0);

    float light_alignment = dot(-dir, light_dir_2d);
    float facing = max(light_alignment, 0.0);
    float away = max(-light_alignment, 0.0);
    float edge = pow(1.0 - depth, 3.0);
    float rim_light = facing * edge * 1.5;
    float rim_shadow = away * edge * 0.35;

    float glow = spec * 0.85 + rim_light;

    // --- Compose ---
    vec4 noise = vec4(vec3(rand(uvec2(gl_FragCoord.xy)) - 0.5), 0.0);
    vec4 color = texture(niri_color, uv) + noise * u_noise;
    color.rgb *= 1.0 + glow - rim_shadow;

    frag_color = color;
}

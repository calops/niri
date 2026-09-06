#version 300 es

precision highp float;
precision highp int;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;

out vec4 frag_color;

// --- CRT knobs ---
// Screen curvature: 0.0 = flat, 0.3 = heavily curved
// Subtle 0.05 | Balanced 0.15 | Heavy 0.3
const float u_curvature = 0.3;

// Chromatic aberration: 0.0 = none, 0.1 = heavy fringe
// Subtle 0.002 | Balanced 0.005 | Heavy 0.008
const float u_chromatic = 0.008;

// Scanline strength: 0.0 = none, 1.0 = full dark lines
// Subtle 0.15 | Balanced 0.35 | Heavy 0.6
const float u_scanlineStrength = 0.6;

// Scanline gap width: 0.25 = thin lines (wide gap), 0.5 = thick lines
// Subtle 0.5 | Balanced 0.45 | Heavy 0.35
const float u_scanlineWidth = 0.35;

// Phosphor mask strength: 0.0 = off, 1.0 = strong dots
// Subtle 0.0 | Balanced 0.3 | Heavy 0.6
const float u_phosphorStrength = 0.6;

// Phosphor dot spacing in output pixels (larger = bigger visible dots)
// Subtle 1.5 | Balanced 2.5 | Heavy 5.0
const float u_phosphorScale = 5.0;

// Bloom strength: 0.0 = none, 1.0 = heavy glow
// Subtle 0.05 | Balanced 0.15 | Heavy 0.3
const float u_bloomStrength = 0.3;

// Vignette strength: 0.0 = none, 1.0 = black corners
// Subtle 0.1 | Balanced 0.3 | Heavy 0.5
const float u_vignetteStrength = 0.5;

// Noise strength: 0.0 = clean, 1.0 = heavy grain
// Subtle 0.005 | Balanced 0.012 | Heavy 0.025
const float u_noiseStrength = 0.025;

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

    vec2 centered = uv - 0.5;
    float dist = dot(centered, centered);

    // --- Curvature ---
    vec2 warped = centered * (1.0 + u_curvature * dist) + 0.5;

    if (warped.x < 0.0 || warped.x > 1.0 || warped.y < 0.0 || warped.y > 1.0) {
        frag_color = vec4(0.0);
        return;
    }

    // --- Chromatic aberration ---
    float ca = u_chromatic * dist;

    vec4 center_sample = texture(niri_input, warped);
    float cr = texture(niri_input, warped + vec2(ca, 0.0)).r;
    float cg = center_sample.g;
    float cb = texture(niri_input, warped - vec2(ca, 0.0)).b;
    float ca_val = center_sample.a;

    vec4 color = vec4(cr, cg, cb, ca_val);

    // --- Phosphor bloom (cheap in-pass blur) ---
    vec4 bloom = vec4(0.0);
    float bloom_kernel = 4.0 / niri_output_size.y;
    bloom += center_sample * 0.5;
    bloom += texture(niri_input, warped + vec2(bloom_kernel, 0.0)) * 0.125;
    bloom += texture(niri_input, warped - vec2(bloom_kernel, 0.0)) * 0.125;
    bloom += texture(niri_input, warped + vec2(0.0, bloom_kernel)) * 0.125;
    bloom += texture(niri_input, warped - vec2(0.0, bloom_kernel)) * 0.125;
    color.rgb += bloom.rgb * u_bloomStrength;

    float row_period = mod(floor(gl_FragCoord.y), 2.0);
    float scanline = mix(u_scanlineWidth, 1.0, row_period);
    color.rgb *= mix(1.0, scanline, u_scanlineStrength);

    // --- Phosphor mask pattern (RGB triad dots) ---
    vec2 pixel_coord = gl_FragCoord.xy / u_phosphorScale;
    vec2 cell_uv = fract(pixel_coord);

    float r_dot = 1.0 - smoothstep(0.35, 0.45, length(cell_uv - vec2(0.25, 0.5)));
    float g_dot = 1.0 - smoothstep(0.35, 0.45, length(cell_uv - vec2(0.5, 0.5)));
    float b_dot = 1.0 - smoothstep(0.35, 0.45, length(cell_uv - vec2(0.75, 0.5)));

    color.r = mix(color.r, color.r * r_dot, u_phosphorStrength);
    color.g = mix(color.g, color.g * g_dot, u_phosphorStrength);
    color.b = mix(color.b, color.b * b_dot, u_phosphorStrength);

    // --- Vignette ---
    float vignette = 1.0 - dist * u_vignetteStrength * 2.0;
    color.rgb *= max(vignette, 0.0);

    // --- Noise ---
    color.rgb += (rand(uvec2(gl_FragCoord.xy)) - 0.5) * u_noiseStrength;

    frag_color = color;
}

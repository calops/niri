#version 100

precision highp float;

varying vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform int niri_pass;
uniform int niri_pass_count;

void main() {
    vec2 uv = v_coords;
    vec2 pixel = 1.0 / niri_input_size;

    // Chromatic aberration - separate RGB channels
    float aberration = 4.0;
    float r = texture2D(niri_input, uv + vec2(pixel.x * aberration, 0.0)).r;
    float g = texture2D(niri_input, uv).g;
    float b = texture2D(niri_input, uv - vec2(pixel.x * aberration, 0.0)).b;
    float a = texture2D(niri_input, uv).a;
    vec4 color = vec4(r, g, b, a);

    // Top highlight to simulate light refraction
    float highlight = smoothstep(1.0, 0.0, uv.y) * 0.03;

    // Fresnel-like edge glow
    vec2 edge = smoothstep(0.0, 0.25, uv) * smoothstep(0.0, 0.25, 1.0 - uv);
    float fresnel = 1.0 - min(edge.x, edge.y);
    fresnel = pow(fresnel, 2.5) * 0.18;

    // Glass tint (cool blue-white)
    vec3 tint = vec3(0.85, 0.9, 1.0);

    // Combine
    color.rgb = mix(color.rgb, color.rgb * tint, 0.15);
    color.rgb += highlight * vec3(0.8, 0.85, 1.0);
    color.rgb += fresnel * vec3(0.7, 0.8, 1.0);

    gl_FragColor = color;
}

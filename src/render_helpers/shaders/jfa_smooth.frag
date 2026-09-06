#version 300 es

precision highp float;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform vec2 niri_output_size;

out vec4 frag_color;

void main() {
    vec2 uv = v_coords;
    vec2 half_texel = 0.5 / niri_output_size;
    vec4 center = texture(niri_input, uv);

    // With LINEAR sampling, these four half-texel taps form a separable
    // 3x3 tent filter. Only the encoded direction is filtered: distance and
    // exterior coverage remain exact, while one-pixel direction blocks are
    // averaged before the bbox-local result is cached.
    vec2 direction = (
        texture(niri_input, uv + vec2(-half_texel.x, -half_texel.y)).gb
        + texture(niri_input, uv + vec2( half_texel.x, -half_texel.y)).gb
        + texture(niri_input, uv + vec2(-half_texel.x,  half_texel.y)).gb
        + texture(niri_input, uv + vec2( half_texel.x,  half_texel.y)).gb
    ) * 0.25;

    frag_color = vec4(center.r, direction, center.a);
}

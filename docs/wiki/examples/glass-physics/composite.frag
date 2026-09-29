#version 300 es
precision highp float;

in vec2 v_coords;
uniform sampler2D niri_color;

out vec4 frag_color;

void main() {
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0;

    float chromatic_strength = smoothstep(0.3, 1.0, dist) * 0.003;
    vec2 chromatic_offset = normalize(center + 0.001) * chromatic_strength;

    float r = texture(niri_color, v_coords + chromatic_offset).r;
    float g = texture(niri_color, v_coords).g;
    float b = texture(niri_color, v_coords - chromatic_offset).b;
    float a = texture(niri_color, v_coords).a;

    vec4 color = vec4(r, g, b, a);
    float specular = pow(max(0.0, 1.0 - dist), 8.0) * 0.15;
    color.rgb += vec3(specular);

    float rim = smoothstep(0.6, 1.0, dist) * 0.05;
    color.rgb += vec3(rim);
    frag_color = color;
}

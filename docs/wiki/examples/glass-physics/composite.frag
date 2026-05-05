#version 100
precision highp float;

varying vec2 v_coords;
uniform sampler2D niri_input;
uniform vec2 niri_input_size;
uniform vec2 niri_output_size;
uniform vec2 niri_geo_size;

void main() {
    vec2 center = v_coords - 0.5;
    float dist = length(center) * 2.0;

    float chromatic_strength = smoothstep(0.3, 1.0, dist) * 0.003;
    vec2 chromatic_offset = normalize(center + 0.001) * chromatic_strength;

    float r = texture2D(niri_input, v_coords + chromatic_offset).r;
    float g = texture2D(niri_input, v_coords).g;
    float b = texture2D(niri_input, v_coords - chromatic_offset).b;
    float a = texture2D(niri_input, v_coords).a;

    vec4 color = vec4(r, g, b, a);

    float specular = pow(max(0.0, 1.0 - dist), 8.0) * 0.15;
    color.rgb += vec3(specular);

    float rim = smoothstep(0.6, 1.0, dist) * 0.05;
    color.rgb += vec3(rim);

    gl_FragColor = color;
}

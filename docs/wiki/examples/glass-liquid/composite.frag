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

    float chromatic_strength = smoothstep(0.2, 1.0, dist) * 0.004;
    vec2 dir = normalize(center + 0.001) * chromatic_strength;

    float r = texture2D(niri_input, v_coords + dir).r;
    float g = texture2D(niri_input, v_coords).g;
    float b = texture2D(niri_input, v_coords - dir).b;
    float a = texture2D(niri_input, v_coords).a;

    vec4 color = vec4(r, g, b, a);

    float edge_highlight = smoothstep(0.7, 1.0, dist) * 0.2;
    color.rgb += vec3(edge_highlight);

    vec2 light_dir = normalize(vec2(-0.5, -0.5));
    float specular = pow(max(0.0, dot(normalize(center), light_dir)), 16.0) * 0.12;
    color.rgb += vec3(specular);

    color.rgb += vec3(0.01, 0.02, 0.04) * smoothstep(0.4, 1.0, dist);

    gl_FragColor = color;
}

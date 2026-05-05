#version 300 es

in vec2 vert;
out vec2 v_coords;

void main() {
    v_coords = vert;
    // vert goes from 0 to 1; position must be from -1 to 1.
    vec2 position = vert * 2.0 - 1.0;
    gl_Position = vec4(position, 1.0, 1.0);
}

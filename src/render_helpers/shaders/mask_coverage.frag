#version 300 es

precision highp float;

in vec2 v_coords;

// Cursor alpha coverage (premultiplied ARGB), used as the authoritative
// interior/exterior classification for the JFA seed.
uniform sampler2D niri_coverage;
// Size of the bbox-local destination in pixels.
uniform vec2 niri_output_size;
// Silhouette rect in bbox-local pixels: x, y, width, height (GL bottom-left).
uniform vec4 niri_coverage_rect;

out vec4 frag_color;

void main() {
    vec2 pixel = v_coords * niri_output_size;
    vec2 local = pixel - niri_coverage_rect.xy;
    if (local.x < 0.0 || local.y < 0.0
        || local.x >= niri_coverage_rect.z || local.y >= niri_coverage_rect.w) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    vec2 uv = local / niri_coverage_rect.zw;
    // The imported cursor texture is top-left origin; the mask is GL bottom-left.
    uv.y = 1.0 - uv.y;
    frag_color = vec4(texture(niri_coverage, uv).a, 0.0, 0.0, 0.0);
}

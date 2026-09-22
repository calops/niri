#version 300 es

precision highp float;

in vec2 v_coords;

// Cursor alpha coverage (premultiplied ARGB), used outside morphs.
uniform sampler2D niri_coverage;
// Signed distances are negative inside. Each endpoint is sampled through its
// own hotspot-aligned rectangle and normalized texture coordinates.
uniform sampler2D niri_sdf_source;
uniform sampler2D niri_sdf_destination;
uniform float niri_sdf_progress;
uniform int niri_sdf_transition;
uniform int niri_sdf_available;
// xy is the velocity direction, z is the hotspot-anchored stretch fraction.
uniform vec3 niri_motion;
uniform vec2 niri_motion_anchor;
uniform vec2 niri_output_size;
// x, y, width, height in the padded union canvas, GL bottom-left origin.
uniform vec4 niri_coverage_rect;
uniform vec4 niri_sdf_source_rect;
uniform vec4 niri_sdf_destination_rect;

out vec4 frag_color;

float sample_sdf(sampler2D sdf_texture, vec4 rect, vec2 pixel) {
    // Continue the endpoint field outside its rectangle from the nearest
    // boundary sample. local and rect use GL bottom-left pixels; the imported
    // SDF texture uses top-left rows, so flip only after normalizing the
    // clamped local coordinate.
    vec2 local = pixel - rect.xy;
    vec2 clamped_local = clamp(local, vec2(0.0), rect.zw);
    vec2 uv = clamped_local / rect.zw;
    uv.y = 1.0 - uv.y;
    float boundary_sdf = texture(sdf_texture, uv).r;
    return boundary_sdf + length(local - clamped_local);
}

void main() {
    vec2 pixel = v_coords * niri_output_size;
    // Inverse-map only along the motion axis, keeping the hotspot fixed. This
    // deforms the SDF contour itself, including during a named-shape morph.
    float stretch = max(1.0 + niri_motion.z, 0.01);
    vec2 relative = pixel - niri_motion_anchor;
    vec2 motion_pixel = niri_motion_anchor + niri_motion.xy
        * (dot(relative, niri_motion.xy) / stretch)
        + (relative - niri_motion.xy * dot(relative, niri_motion.xy));
    // Axial scaling alone is symmetric, so reversing direction looks identical.
    // Pull samples toward the trailing side by an amount that grows away from
    // the hotspot, giving the elastic contour a visible leading direction.
    motion_pixel += niri_motion.xy * niri_motion.z * 0.4 * length(relative);
    float alpha;
    if (niri_sdf_transition != 0) {
        float source = sample_sdf(niri_sdf_source, niri_sdf_source_rect, motion_pixel);
        float destination = sample_sdf(niri_sdf_destination, niri_sdf_destination_rect, motion_pixel);
        float sdf = mix(source, destination, niri_sdf_progress);
        // A one-pixel contour ramp; this morphs distance fields rather than alpha.
        alpha = smoothstep(0.5, -0.5, sdf);
    } else if (niri_sdf_available != 0) {
        float sdf = sample_sdf(niri_sdf_destination, niri_sdf_destination_rect, motion_pixel);
        alpha = smoothstep(0.5, -0.5, sdf);
    } else {
        vec2 local = pixel - niri_coverage_rect.xy;
        if (local.x < 0.0 || local.y < 0.0 || local.x >= niri_coverage_rect.z || local.y >= niri_coverage_rect.w) {
            frag_color = vec4(0.0);
            return;
        }
        vec2 uv = local / niri_coverage_rect.zw;
        uv.y = 1.0 - uv.y;
        alpha = texture(niri_coverage, uv).a;
    }
    frag_color = vec4(alpha, 0.0, 0.0, 0.0);
}

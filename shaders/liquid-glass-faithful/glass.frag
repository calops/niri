#version 300 es

precision highp float;
precision highp int;

in vec2 v_coords;

uniform sampler2D niri_input;
uniform sampler2D niri_mask;
uniform vec4 niri_mask_uv_rect;
uniform vec2 niri_output_size;
uniform vec2 niri_input_size;
uniform vec2 niri_half_pixel;
uniform vec2 niri_geo_size;
uniform vec4 niri_corner_radius;
uniform vec4 niri_window_screen_rect;

out vec4 frag_color;

const float M_E = 2.718281828459045;

// Physical width of the refractive and illuminated edge in source pixels.
const float u_glassThicknessPx = 32.0;

// Defaults from OverShifted/LiquidGlass. The original exposes these through
// ImGui; keeping them literal makes this shader's transfer function identical.
const float u_a = 0.7;
const float u_b = 2.3;
const float u_c = 5.2;
const float u_d = 6.9;
const float u_fPower = 1.0;
const float u_noise = 0.065;
const float u_glowWeight = 0.48;
const float u_glowBias = 0.0;
const float u_glowEdge0 = 0.18;
const float u_glowEdge1 = 0.0;
const float u_neutralHighlight = 0.16;

// Broad screen-space source for edge illumination. Coordinates are output UV:
// left edge = 0, bottom edge = 0. The source sits left and one-third from top.
const vec2 u_lightPos = vec2(0.12, 0.67);
const float u_lightRadius = 0.55;
const float u_lightFloor = 0.55;

// Separate rough specular reflection from the same broad source. The surface
// normal tilts only through the lens edge and remains flat on the plateau.
const float u_specularStrength = 0.65;
const float u_specularPower = 8.0;
const float u_edgeSlope = 3.4;
const vec3 u_specularColor = vec3(0.97, 0.985, 1.0);

// Restrained radial dispersion across the flat face, with a stronger component
// following the refraction curve near the edge.
const float u_chromaticBase = 0.0025;
const float u_chromaticEdge = 0.035;

float refractionCurve(float x) {
    return 1.0 - u_b * pow(u_c * M_E, -u_d * x - u_a);
}

float cornerRadius(vec2 p) {
    if (p.y < 0.0) {
        return p.x < 0.0 ? niri_corner_radius.x : niri_corner_radius.y;
    }
    return p.x < 0.0 ? niri_corner_radius.w : niri_corner_radius.z;
}

vec2 roundedRectOutward(vec2 pixel) {
    vec2 half_size = niri_geo_size * 0.5;
    vec2 p = pixel - half_size;
    float geometry_radius = min(cornerRadius(p), min(half_size.x, half_size.y));

    // A square silhouette has no radial corner field, so its exact
    // rounded-rectangle gradient exposes a nearest-edge split from each corner.
    // Give lighting alone enough optical rounding to move that split beyond the
    // active glow. The mask, clipping, and refraction still use the configured
    // geometry radius.
    float glow_width = min(u_glassThicknessPx, min(half_size.x, half_size.y));
    float radius = max(geometry_radius, glow_width);
    vec2 q = abs(p) - half_size + radius;

    // Smooth the remaining medial transition while retaining radial normals
    // across the optically rounded corner.
    float softness = clamp(radius * 0.25, 2.0, 12.0);
    float x_weight = smoothstep(-softness, softness, q.x - q.y);
    vec2 medial = normalize(vec2(x_weight, 1.0 - x_weight));
    vec2 corner_vector = max(q, 0.0);
    float corner_length = length(corner_vector);
    vec2 corner = corner_length > 1e-5 ? corner_vector / corner_length : medial;
    float corner_weight = smoothstep(0.0, softness, min(q.x, q.y));
    vec2 local_outward = normalize(mix(medial, corner, corner_weight));

    return local_outward * sign(p);
}

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = v_coords;
    vec2 mask_uv = mix(niri_mask_uv_rect.xy, niri_mask_uv_rect.zw, uv);
    vec4 mask_sample = texture(niri_mask, mask_uv);
    float dist = mask_sample.r;

    if (dist < 0.001) {
        frag_color = texture(niri_input, uv);
        return;
    }

    // window-vectors normalizes distance by half the shorter geometry axis.
    // Remap it to the original curve's domain through a fixed pixel width so
    // differently sized windows retain the same apparent glass thickness.
    float min_half_size = 0.5 * min(niri_geo_size.x, niri_geo_size.y);
    float distance_px = dist * min_half_size;
    float optical_dist = distance_px * u_glowEdge0 / u_glassThicknessPx;

    // The original computes p in local object coordinates, scales p radially by
    // pow(f(distance), power), then maps it back into screen space. niri's GB
    // channels give the exact vector from this fragment to the window centre,
    // so p * scale becomes uv + to_center * (1 - scale).
    vec2 to_center = (mask_sample.gb - 0.5) * 2.0;
    float scale = pow(max(refractionCurve(optical_dist), 0.0), u_fPower);
    float displacement = 1.0 - scale;

    vec2 uv_g = clamp(uv + to_center * displacement, 0.0, 1.0);
    float chromatic_strength = u_chromaticBase + displacement * u_chromaticEdge;
    vec2 chromatic_offset = to_center * chromatic_strength;
    vec2 uv_r = clamp(uv_g - chromatic_offset, 0.0, 1.0);
    vec2 uv_b = clamp(uv_g + chromatic_offset, 0.0, 1.0);

    vec4 green_alpha = texture(niri_input, uv_g);
    vec4 color = vec4(
        texture(niri_input, uv_r).r,
        green_alpha.g,
        texture(niri_input, uv_b).b,
        green_alpha.a
    );

    // Preserve the original broad edge-light response, but anchor its source in
    // output space so moving the window changes which edge faces the light.
    // This is diffuse edge illumination, intentionally separate from any future
    // specular reflection term.
    vec2 outward = roundedRectOutward(mask_uv * niri_geo_size);
    vec2 screen_uv = niri_window_screen_rect.xy
        + mask_uv * niri_window_screen_rect.zw;
    vec2 to_light_field = u_lightPos - screen_uv;
    float light_distance = length(to_light_field);
    vec2 to_light = light_distance > 1e-5
        ? to_light_field / light_distance
        : vec2(0.0);
    float light_falloff = mix(
        u_lightFloor,
        1.0,
        exp(-pow(light_distance / u_lightRadius, 2.0))
    );
    float glow = dot(outward, to_light) * light_falloff;
    float glow_mask = smoothstep(u_glowEdge0, u_glowEdge1, optical_dist);
    float specular_mask = smoothstep(0.085, 0.0, optical_dist);
    float highlight = max(glow, 0.0) * glow_mask;
    float shadow = max(-glow, 0.0) * glow_mask;

    // Rough Blinn-Phong reflection from the same screen-space source. This is
    // independent from the diffuse edge illumination above: the highlight
    // appears only where the curved edge normal reflects the source toward the
    // viewer. The smoothed analytical field avoids nearest-edge ownership seams.
    vec3 surface_normal = normalize(vec3(outward * u_edgeSlope * specular_mask, 1.0));
    vec3 light_direction = normalize(vec3(to_light_field, 0.42));
    vec3 view_direction = vec3(0.0, 0.0, 1.0);
    vec3 half_direction = normalize(light_direction + view_direction);
    float specular = pow(max(dot(surface_normal, half_direction), 0.0), u_specularPower);
    specular *= specular_mask * light_falloff * u_specularStrength;

    vec3 noise = vec3(rand(gl_FragCoord.xy * 1e-3) - 0.5) * u_noise;
    color.rgb += noise;

    color.rgb *= 1.0 - shadow * u_glowWeight + u_glowBias;
    color.rgb += vec3(color.a) * highlight * u_glowWeight * u_neutralHighlight;
    color.rgb *= 1.0 + highlight * u_glowWeight * (1.0 - u_neutralHighlight);
    color.rgb += u_specularColor * color.a * specular;
    frag_color = color;
}

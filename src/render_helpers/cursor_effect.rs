use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use niri_config::ShaderPipeline;
use smithay::input::pointer::CursorIcon;
use smithay::utils::{Buffer, Logical, Point, Rectangle, Scale, Size};

use crate::render_helpers::background_effect::RenderParams;
use crate::render_helpers::blur::{BlurOptions, CoverageMask};
use crate::render_helpers::framebuffer_effect::{FramebufferEffect, FramebufferEffectElement};

/// Per-Renderer state for rendering the cursor through the shader pipeline.
///
/// Mirrors `BackgroundEffect`, but for the cursor: the silhouette mask is
/// tightly bounded and static per cursor frame, while the effect region is
/// padded so refraction can sample the backdrop beyond the silhouette.
#[derive(Debug)]
pub struct CursorEffect {
    effect: FramebufferEffect,
    /// Identity of the cursor frame currently reflected in the cached mask.
    /// Used to damage the output when the cursor image itself changes, since
    /// the element geometry alone does not change then.
    last_identity: Option<u64>,
}

/// The padded effect region and the silhouette placement within it.
#[derive(Debug, Clone, Copy)]
pub struct CursorEffectGeometry {
    /// Padded effect region in output-local logical coordinates.
    pub geometry: Rectangle<f64, Logical>,
    /// Silhouette rect within the mask, in mask pixels (GL bottom-left).
    pub coverage_bbox: Rectangle<i32, Buffer>,
}

impl CursorEffect {
    pub fn new() -> Self {
        Self {
            effect: FramebufferEffect::new(),
            last_identity: None,
        }
    }

    /// Build the framebuffer-effect element for the cursor silhouette.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        ns: Option<usize>,
        geometry: CursorEffectGeometry,
        scale: f64,
        coverage: CoverageMask,
        passes: u8,
        offset: f64,
        pipeline: ShaderPipeline,
    ) -> FramebufferEffectElement {
        if self.last_identity != Some(coverage.identity) {
            self.effect.damage();
            self.last_identity = Some(coverage.identity);
            debug!(
                "cursor effect: rendering shader, region {:?}, silhouette {:?}",
                geometry.geometry, geometry.coverage_bbox
            );
        }

        let params = RenderParams {
            geometry: geometry.geometry,
            subregion: None,
            clip: None,
            scale,
        };
        let options = BlurOptions {
            passes,
            offset,
            geo_size: (
                geometry.geometry.size.w as f32,
                geometry.geometry.size.h as f32,
            ),
            corner_radius: [0.0; 4],
            subregion_rects: Arc::new(Vec::new()),
            // Overwritten by FramebufferEffectElement from the screen-space
            // geometry, which keeps the shader light screen-anchored.
            window_screen_rect: [0.0; 4],
            mask_size: None,
            mask_uv_rect: [0.0, 0.0, 1.0, 1.0],
            shader_pipeline: Some(pipeline),
            coverage_mask: Some(coverage),
        };

        self.effect.render(ns, params, Some(options), 0.0, 1.0)
    }
}

/// Compute the padded cursor effect region and the silhouette placement in
/// mask pixels (GL bottom-left origin).
pub fn cursor_effect_geometry(
    pointer_pos: Point<f64, Logical>,
    hotspot: Point<f64, Logical>,
    frame_size: Size<f64, Logical>,
    padding: f64,
    scale: Scale<f64>,
) -> CursorEffectGeometry {
    let silhouette = Rectangle::new(pointer_pos - hotspot, frame_size);
    // At least one pixel of margin, otherwise the JFA bbox clamps against the
    // mask edge and loses its exterior seed ring.
    let padding = padding.max(1.0);
    let geometry = Rectangle::new(
        Point::from((silhouette.loc.x - padding, silhouette.loc.y - padding)),
        Size::from((
            (silhouette.size.w + 2.0 * padding).max(1.0),
            (silhouette.size.h + 2.0 * padding).max(1.0),
        )),
    );

    let geo_phys = geometry.to_physical_precise_round(scale);
    let sil_phys = silhouette.to_physical_precise_round(scale);
    let rel = sil_phys.loc - geo_phys.loc;
    let coverage_bbox = Rectangle::new(
        Point::from((rel.x, geo_phys.size.h - rel.y - sil_phys.size.h)),
        Size::from((sil_phys.size.w, sil_phys.size.h)),
    );

    CursorEffectGeometry {
        geometry,
        coverage_bbox,
    }
}

/// Identity of a cursor frame, used as the mask/JFA cache key. Deliberately
/// excludes position so that moving the cursor reuses the computed mask.
pub fn cursor_frame_identity(
    icon: CursorIcon,
    scale: i32,
    idx: usize,
    width: u32,
    height: u32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    icon.name().hash(&mut hasher);
    scale.hash(&mut hasher);
    idx.hash(&mut hasher);
    width.hash(&mut hasher);
    height.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_widens_the_region_and_insets_the_silhouette() {
        let geo = cursor_effect_geometry(
            Point::from((100.0, 100.0)),
            Point::from((1.0, 1.0)),
            Size::from((10.0, 20.0)),
            5.0,
            Scale::from(1.0),
        );
        assert_eq!(geo.geometry.loc, Point::from((94.0, 94.0)));
        assert_eq!(geo.geometry.size, Size::from((20.0, 30.0)));
        // Mask pixels are GL bottom-left, so a symmetric padding insets the
        // silhouette equally on every side.
        assert_eq!(geo.coverage_bbox.loc, Point::from((5, 5)));
        assert_eq!(geo.coverage_bbox.size, Size::from((10, 20)));
    }

    #[test]
    fn padding_scales_with_the_output() {
        let geo = cursor_effect_geometry(
            Point::from((100.0, 100.0)),
            Point::from((1.0, 1.0)),
            Size::from((10.0, 20.0)),
            5.0,
            Scale::from(2.0),
        );
        assert_eq!(geo.coverage_bbox.loc, Point::from((10, 10)));
        assert_eq!(geo.coverage_bbox.size, Size::from((20, 40)));
    }

    #[test]
    fn frame_identity_tracks_the_frame_but_not_position() {
        let a = cursor_frame_identity(CursorIcon::Default, 2, 0, 32, 32);
        assert_eq!(a, cursor_frame_identity(CursorIcon::Default, 2, 0, 32, 32));
        assert_ne!(a, cursor_frame_identity(CursorIcon::Default, 2, 1, 32, 32));
        assert_ne!(a, cursor_frame_identity(CursorIcon::Text, 2, 0, 32, 32));
    }
}

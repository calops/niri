use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use niri_config::ShaderPipeline;
use smithay::backend::renderer::gles::GlesTexture;
use smithay::input::pointer::CursorIcon;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size};

use crate::render_helpers::background_effect::RenderParams;
use crate::render_helpers::blur::{BlurOptions, CoverageMask, SdfTransition};
use crate::render_helpers::framebuffer_effect::{FramebufferEffect, FramebufferEffectElement};

/// Per-renderer state for rendering cursor shader effects. State is separate for
/// every output namespace because each output has its own framebuffer cache and
/// shape-transition timeline.
#[derive(Debug, Default)]
pub struct CursorEffect {
    outputs: HashMap<Option<usize>, CursorEffectState>,
}

#[derive(Debug)]
struct CursorEffectState {
    effect: FramebufferEffect,
    last_identity: Option<u64>,
    last_named_identity: Option<u64>,
    last_sdf: Option<GlesTexture>,
    last_geometry: Option<CursorEffectGeometry>,
    transition: Option<CursorTransition>,
    force_transition: bool,
}

#[derive(Debug)]
struct CursorTransition {
    source: GlesTexture,
    destination: GlesTexture,
    source_identity: u64,
    destination_identity: u64,
    destination_named_identity: u64,
    source_geometry: CursorEffectGeometry,
    destination_geometry: CursorEffectGeometry,
    started_at: Duration,
    duration: Duration,
}

/// The padded effect region and silhouette placement. Physical metadata lets a
/// transition rebuild both endpoint placements around the current pointer.
#[derive(Debug, Clone, Copy)]
pub struct CursorEffectGeometry {
    pub geometry: Rectangle<f64, Logical>,
    pub coverage_bbox: Rectangle<i32, Buffer>,
    pointer: Point<i32, Physical>,
    hotspot: Point<i32, Physical>,
    frame_size: Size<i32, Physical>,
    padding: i32,
    scale: Scale<f64>,
}

impl CursorEffect {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        ns: Option<usize>,
        now: Duration,
        geometry: CursorEffectGeometry,
        scale: f64,
        coverage: CoverageMask,
        shape_transition_duration_ms: u32,
        passes: u8,
        offset: f64,
        pipeline: ShaderPipeline,
    ) -> FramebufferEffectElement {
        let state = self.outputs.entry(ns).or_insert_with(|| CursorEffectState {
            effect: FramebufferEffect::new(),
            last_identity: None,
            last_named_identity: None,
            last_sdf: None,
            last_geometry: None,
            transition: None,
            force_transition: false,
        });
        let (geometry, coverage) =
            state.transition_coverage(coverage, geometry, now, shape_transition_duration_ms);
        if state.last_identity != Some(coverage.identity) || coverage.sdf_transition.is_some() {
            // Intermediate progress changes the mask every frame, even when its
            // element geometry is unchanged.
            state.effect.damage();
            state.last_identity = Some(coverage.identity);
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
            window_screen_rect: [0.0; 4],
            mask_size: None,
            mask_uv_rect: [0.0, 0.0, 1.0, 1.0],
            shader_pipeline: Some(pipeline),
            coverage_mask: Some(coverage),
        };
        state.effect.render(ns, params, Some(options), 0.0, 1.0)
    }

    /// Preserve endpoint SDFs while making the next named cursor render morph
    /// to freshly loaded cursor assets.
    pub fn reload(&mut self) {
        for state in self.outputs.values_mut() {
            state.force_transition = state.last_sdf.is_some();
        }
    }

    pub fn transition_active(&self, now: Duration) -> bool {
        self.outputs.values().any(|state| {
            state
                .transition
                .as_ref()
                .is_some_and(|transition| now < transition.started_at + transition.duration)
        })
    }
}

impl CursorEffectState {
    fn transition_coverage(
        &mut self,
        mut coverage: CoverageMask,
        geometry: CursorEffectGeometry,
        now: Duration,
        duration_ms: u32,
    ) -> (CursorEffectGeometry, CoverageMask) {
        let Some(named_identity) = coverage.named_identity else {
            self.last_sdf = coverage.sdf.clone();
            self.last_geometry = Some(geometry);
            return (geometry, coverage);
        };
        let changed_shape = self.force_transition
            || self.transition.as_ref().map_or_else(
                || {
                    self.last_named_identity
                        .is_some_and(|old| old != named_identity)
                },
                // Keep an in-flight transition alive when xcursor advances a
                // frame of the same named shape; only its destination SDF is
                // retained, never replaced by that animation frame.
                |_| true,
            );
        self.last_named_identity = Some(named_identity);
        self.force_transition = false;
        if duration_ms == 0 || !changed_shape {
            self.transition = None;
            self.last_sdf = coverage.sdf.clone();
            self.last_geometry = Some(geometry);
            return (geometry, coverage);
        }
        let Some(destination) = coverage.sdf.clone() else {
            return (geometry, coverage);
        };

        if self
            .transition
            .as_ref()
            .is_none_or(|transition| transition.destination_named_identity != named_identity)
        {
            let (source, source_identity, source_geometry) = self.transition.take().map_or_else(
                || {
                    (
                        self.last_sdf.clone().unwrap_or_else(|| destination.clone()),
                        coverage.identity,
                        self.last_geometry.unwrap_or(geometry),
                    )
                },
                |transition| {
                    (
                        transition.destination,
                        transition.destination_identity,
                        transition.destination_geometry,
                    )
                },
            );
            self.transition = Some(CursorTransition {
                source,
                destination,
                source_identity,
                destination_identity: coverage.identity,
                destination_named_identity: named_identity,
                source_geometry,
                destination_geometry: geometry,
                started_at: now,
                duration: Duration::from_millis(duration_ms.into()),
            });
        }
        let transition = self.transition.as_ref().unwrap();
        let progress = ((now - transition.started_at).as_secs_f32()
            / transition.duration.as_secs_f32())
        .min(1.0);
        let union = cursor_effect_union_geometry(
            transition.source_geometry,
            transition.destination_geometry,
            geometry.pointer,
        );
        // The JFA/coverage target is the full union canvas: both endpoint
        // contours must remain available throughout the morph.
        coverage.bbox = union.geometry.coverage_bbox;
        coverage.sdf_transition = Some(SdfTransition {
            source: transition.source.clone(),
            destination: transition.destination.clone(),
            source_rect: union.source_rect,
            destination_rect: union.destination_rect,
            progress,
        });
        coverage.identity = transition_identity(
            transition.source_identity,
            transition.destination_identity,
            union.source_rect,
            union.destination_rect,
            progress,
        );
        if progress >= 1.0 {
            self.last_sdf = Some(transition.destination.clone());
            self.last_geometry = Some(geometry);
            self.transition = None;
        }
        (union.geometry, coverage)
    }
}

pub fn cursor_effect_geometry(
    pointer_pos: Point<f64, Logical>,
    hotspot: Point<f64, Logical>,
    frame_size: Size<f64, Logical>,
    padding: f64,
    scale: Scale<f64>,
) -> CursorEffectGeometry {
    let pointer = pointer_pos.to_physical_precise_round(scale);
    let hotspot = hotspot.to_physical_precise_round(scale);
    let frame_size = frame_size.to_physical_precise_round(scale);
    let padding = (padding.max(1.0) * scale.x).round() as i32;
    cursor_effect_geometry_physical(pointer, hotspot, frame_size, padding, scale)
}

fn cursor_effect_geometry_physical(
    pointer: Point<i32, Physical>,
    hotspot: Point<i32, Physical>,
    frame_size: Size<i32, Physical>,
    padding: i32,
    scale: Scale<f64>,
) -> CursorEffectGeometry {
    let silhouette = Rectangle::new(pointer - hotspot, frame_size);
    let geo_phys = Rectangle::new(
        Point::from((silhouette.loc.x - padding, silhouette.loc.y - padding)),
        Size::from((
            silhouette.size.w + 2 * padding,
            silhouette.size.h + 2 * padding,
        )),
    );
    let rel = silhouette.loc - geo_phys.loc;
    let coverage_bbox = Rectangle::<i32, Buffer>::new(
        Point::from((rel.x, geo_phys.size.h - rel.y - silhouette.size.h)),
        Size::from((silhouette.size.w, silhouette.size.h)),
    );
    CursorEffectGeometry {
        geometry: geo_phys.to_f64().to_logical(scale),
        coverage_bbox,
        pointer,
        hotspot,
        frame_size,
        padding,
        scale,
    }
}

struct CursorEffectUnionGeometry {
    geometry: CursorEffectGeometry,
    source_rect: Rectangle<i32, Buffer>,
    destination_rect: Rectangle<i32, Buffer>,
}

fn cursor_effect_union_geometry(
    source: CursorEffectGeometry,
    destination: CursorEffectGeometry,
    pointer: Point<i32, Physical>,
) -> CursorEffectUnionGeometry {
    let source = cursor_effect_geometry_physical(
        pointer,
        source.hotspot,
        source.frame_size,
        source.padding,
        source.scale,
    );
    let destination = cursor_effect_geometry_physical(
        pointer,
        destination.hotspot,
        destination.frame_size,
        destination.padding,
        destination.scale,
    );
    let source_silhouette = Rectangle::new(pointer - source.hotspot, source.frame_size);
    let destination_silhouette =
        Rectangle::new(pointer - destination.hotspot, destination.frame_size);
    let min_x = source_silhouette.loc.x.min(destination_silhouette.loc.x)
        - source.padding.max(destination.padding);
    let min_y = source_silhouette.loc.y.min(destination_silhouette.loc.y)
        - source.padding.max(destination.padding);
    let max_x = (source_silhouette.loc.x + source_silhouette.size.w)
        .max(destination_silhouette.loc.x + destination_silhouette.size.w)
        + source.padding.max(destination.padding);
    let max_y = (source_silhouette.loc.y + source_silhouette.size.h)
        .max(destination_silhouette.loc.y + destination_silhouette.size.h)
        + source.padding.max(destination.padding);
    let union_phys = Rectangle::<i32, Physical>::new(
        Point::from((min_x, min_y)),
        Size::from((max_x - min_x, max_y - min_y)),
    );
    let rect = |silhouette: Rectangle<i32, Physical>| {
        Rectangle::<i32, Buffer>::new(
            Point::from((
                silhouette.loc.x - min_x,
                max_y - silhouette.loc.y - silhouette.size.h,
            )),
            Size::from((silhouette.size.w, silhouette.size.h)),
        )
    };
    CursorEffectUnionGeometry {
        geometry: CursorEffectGeometry {
            geometry: union_phys.to_f64().to_logical(destination.scale),
            // The coverage/JFA mask spans the entire union canvas; endpoint
            // silhouette rectangles below remain relative to this canvas.
            coverage_bbox: Rectangle::<i32, Buffer>::from_size(Size::from((
                union_phys.size.w,
                union_phys.size.h,
            ))),
            pointer,
            hotspot: destination.hotspot,
            frame_size: destination.frame_size,
            padding: destination.padding,
            scale: destination.scale,
        },
        source_rect: rect(source_silhouette),
        destination_rect: rect(destination_silhouette),
    }
}

fn transition_identity(
    source: u64,
    destination: u64,
    source_rect: Rectangle<i32, Buffer>,
    destination_rect: Rectangle<i32, Buffer>,
    progress: f32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    destination.hash(&mut hasher);
    source_rect.loc.x.hash(&mut hasher);
    source_rect.loc.y.hash(&mut hasher);
    source_rect.size.w.hash(&mut hasher);
    source_rect.size.h.hash(&mut hasher);
    destination_rect.loc.x.hash(&mut hasher);
    destination_rect.loc.y.hash(&mut hasher);
    destination_rect.size.w.hash(&mut hasher);
    destination_rect.size.h.hash(&mut hasher);
    progress.to_bits().hash(&mut hasher);
    hasher.finish()
}

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
        assert_eq!(geo.coverage_bbox.loc, Point::from((5, 5)));
        assert_eq!(geo.coverage_bbox.size, Size::from((10, 20)));
    }

    #[test]
    fn union_aligns_different_hotspots_and_frame_sizes() {
        let source = cursor_effect_geometry(
            Point::from((100.0, 100.0)),
            Point::from((2.0, 4.0)),
            Size::from((12.0, 8.0)),
            3.0,
            Scale::from(1.0),
        );
        let destination = cursor_effect_geometry(
            Point::from((100.0, 100.0)),
            Point::from((6.0, 1.0)),
            Size::from((20.0, 16.0)),
            3.0,
            Scale::from(1.0),
        );
        let union = cursor_effect_union_geometry(source, destination, Point::from((100, 100)));
        assert_eq!(union.geometry.geometry.loc, Point::from((91.0, 93.0)));
        assert_eq!(union.geometry.geometry.size, Size::from((26.0, 25.0)));
        assert_eq!(
            union.geometry.coverage_bbox,
            Rectangle::from_size(Size::from((26, 25)))
        );
        assert_eq!(
            union.source_rect,
            Rectangle::new(Point::from((7, 14)), Size::from((12, 8)))
        );
        assert_eq!(
            union.destination_rect,
            Rectangle::new(Point::from((3, 3)), Size::from((20, 16)))
        );
    }
}

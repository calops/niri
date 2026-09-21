use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::rc::Rc;

use anyhow::{anyhow, Context};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::memory::MemoryRenderBuffer;
use smithay::backend::renderer::gles::{ffi, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{ContextId, ImportMem, Renderer};
use smithay::input::pointer::{CursorIcon, CursorImageStatus, CursorImageSurfaceData};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{IsAlive, Logical, Physical, Point, Transform};
use smithay::wayland::compositor::with_states;
use xcursor::parser::{parse_xcursor, Image};
use xcursor::CursorTheme;

/// Some default looking `left_ptr` icon.
static FALLBACK_CURSOR_DATA: &[u8] = include_bytes!("../resources/cursor.rgba");

type XCursorCache = HashMap<(CursorIcon, i32), Option<Rc<XCursor>>>;

pub struct CursorManager {
    theme: CursorTheme,
    size: u8,
    current_cursor: CursorImageStatus,
    named_cursor_cache: RefCell<XCursorCache>,
}

impl CursorManager {
    pub fn new(theme: &str, size: u8) -> Self {
        Self::ensure_env(theme, size);

        let theme = CursorTheme::load(theme);

        Self {
            theme,
            size,
            current_cursor: CursorImageStatus::default_named(),
            named_cursor_cache: Default::default(),
        }
    }

    /// Reload the cursor theme.
    pub fn reload(&mut self, theme: &str, size: u8) {
        Self::ensure_env(theme, size);
        self.theme = CursorTheme::load(theme);
        self.size = size;
        self.named_cursor_cache.get_mut().clear();
    }

    /// Checks if the cursor WlSurface is alive, and if not, cleans it up.
    pub fn check_cursor_image_surface_alive(&mut self) {
        if let CursorImageStatus::Surface(surface) = &self.current_cursor {
            if !surface.alive() {
                self.current_cursor = CursorImageStatus::default_named();
            }
        }
    }

    /// Get the current rendering cursor.
    pub fn get_render_cursor(&self, scale: i32) -> RenderCursor {
        match self.current_cursor.clone() {
            CursorImageStatus::Hidden => RenderCursor::Hidden,
            CursorImageStatus::Surface(surface) => {
                let hotspot = with_states(&surface, |states| {
                    states
                        .data_map
                        .get::<CursorImageSurfaceData>()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .hotspot
                });

                RenderCursor::Surface { hotspot, surface }
            }
            CursorImageStatus::Named(icon) => self.get_render_cursor_named(icon, scale),
        }
    }

    fn get_render_cursor_named(&self, icon: CursorIcon, scale: i32) -> RenderCursor {
        self.get_cursor_with_name(icon, scale)
            .map(|cursor| RenderCursor::Named {
                icon,
                scale,
                cursor,
            })
            .unwrap_or_else(|| RenderCursor::Named {
                icon: Default::default(),
                scale,
                cursor: self.get_default_cursor(scale),
            })
    }

    pub fn is_current_cursor_animated(&self, scale: i32) -> bool {
        match &self.current_cursor {
            CursorImageStatus::Hidden => false,
            CursorImageStatus::Surface(_) => false,
            CursorImageStatus::Named(icon) => self
                .get_cursor_with_name(*icon, scale)
                .unwrap_or_else(|| self.get_default_cursor(scale))
                .is_animated_cursor(),
        }
    }

    /// Get named cursor for the given `icon` and `scale`.
    pub fn get_cursor_with_name(&self, icon: CursorIcon, scale: i32) -> Option<Rc<XCursor>> {
        self.named_cursor_cache
            .borrow_mut()
            .entry((icon, scale))
            .or_insert_with_key(|(icon, scale)| {
                let size = self.size as i32 * scale;
                let mut cursor = Self::load_xcursor(&self.theme, icon.name(), size);

                // Check alternative names to account for non-compliant themes.
                if cursor.is_err() {
                    for name in icon.alt_names() {
                        cursor = Self::load_xcursor(&self.theme, name, size);
                        if cursor.is_ok() {
                            break;
                        }
                    }
                }

                if let Err(err) = &cursor {
                    warn!("error loading xcursor {}@{size}: {err:?}", icon.name());
                }

                // The default cursor must always have a fallback.
                if *icon == CursorIcon::Default && cursor.is_err() {
                    cursor = Ok(Self::fallback_cursor());
                }

                cursor.ok().map(Rc::new)
            })
            .clone()
    }

    /// Get default cursor.
    pub fn get_default_cursor(&self, scale: i32) -> Rc<XCursor> {
        // The default cursor always has a fallback.
        self.get_cursor_with_name(CursorIcon::Default, scale)
            .unwrap()
    }

    /// Currently used cursor_image as a cursor provider.
    pub fn cursor_image(&self) -> &CursorImageStatus {
        &self.current_cursor
    }

    /// Set new cursor image provider.
    pub fn set_cursor_image(&mut self, cursor: CursorImageStatus) {
        match &cursor {
            CursorImageStatus::Hidden => debug!("cursor image: hidden"),
            CursorImageStatus::Surface(_) => {
                debug!("cursor image: client surface (cursor shader bypassed)")
            }
            CursorImageStatus::Named(icon) => debug!("cursor image: named {icon:?}"),
        }
        self.current_cursor = cursor;
    }

    /// Load the cursor with the given `name` from the file system picking the closest
    /// one to the given `size`.
    fn load_xcursor(theme: &CursorTheme, name: &str, size: i32) -> anyhow::Result<XCursor> {
        let _span = tracy_client::span!("load_xcursor");

        let path = theme
            .load_icon(name)
            .ok_or_else(|| anyhow!("no default icon"))?;

        let mut file = File::open(path).context("error opening cursor icon file")?;
        let mut buf = vec![];
        file.read_to_end(&mut buf)
            .context("error reading cursor icon file")?;

        let mut images = parse_xcursor(&buf).context("error parsing cursor icon file")?;

        let (width, height) = images
            .iter()
            .min_by_key(|image| (size - image.size as i32).abs())
            .map(|image| (image.width, image.height))
            .unwrap();

        images.retain(move |image| image.width == width && image.height == height);

        let animation_duration = images.iter().fold(0, |acc, image| acc + image.delay);

        Ok(XCursor {
            images,
            animation_duration,
        })
    }

    /// Set the common XCURSOR env variables.
    fn ensure_env(theme: &str, size: u8) {
        env::set_var("XCURSOR_THEME", theme);
        env::set_var("XCURSOR_SIZE", size.to_string());
    }

    fn fallback_cursor() -> XCursor {
        let images = vec![Image {
            size: 32,
            width: 64,
            height: 64,
            xhot: 1,
            yhot: 1,
            delay: 0,
            pixels_rgba: Vec::from(FALLBACK_CURSOR_DATA),
            pixels_argb: vec![],
        }];

        XCursor {
            images,
            animation_duration: 0,
        }
    }
}

/// The cursor prepared for renderer.
pub enum RenderCursor {
    Hidden,
    Surface {
        hotspot: Point<i32, Logical>,
        surface: WlSurface,
    },
    Named {
        icon: CursorIcon,
        scale: i32,
        cursor: Rc<XCursor>,
    },
}

type TextureCache = HashMap<(CursorIcon, i32), Vec<MemoryRenderBuffer>>;
type CoverageCache = HashMap<(CursorIcon, i32, usize), (ContextId<GlesTexture>, GlesTexture)>;
type SdfCache = HashMap<(CursorIcon, i32, usize), (ContextId<GlesTexture>, GlesTexture)>;

#[derive(Default)]
pub struct CursorTextureCache {
    cache: RefCell<TextureCache>,
    /// Cursor alpha coverage textures for the shader pipeline mask, keyed by
    /// icon, scale, and animation frame.
    coverage: RefCell<CoverageCache>,
    /// Signed-distance textures (negative inside) used to morph named cursor shapes.
    sdf: RefCell<SdfCache>,
}

impl CursorTextureCache {
    pub fn clear(&mut self) {
        self.cache.get_mut().clear();
        self.coverage.get_mut().clear();
        self.sdf.get_mut().clear();
    }

    pub fn get(
        &self,
        icon: CursorIcon,
        scale: i32,
        cursor: &XCursor,
        idx: usize,
    ) -> MemoryRenderBuffer {
        self.cache
            .borrow_mut()
            .entry((icon, scale))
            .or_insert_with(|| {
                cursor
                    .frames()
                    .iter()
                    .map(|frame| {
                        MemoryRenderBuffer::from_slice(
                            &frame.pixels_rgba,
                            Fourcc::Argb8888,
                            (frame.width as i32, frame.height as i32),
                            scale,
                            Transform::Normal,
                            None,
                        )
                    })
                    .collect()
            })[idx]
            .clone()
    }

    /// Get the cursor frame as an alpha coverage texture for the shader
    /// pipeline mask. Xcursor pixels are premultiplied ARGB, so the alpha
    /// channel is the silhouette coverage. The texture is re-imported if the
    /// renderer context changed.
    pub fn get_coverage(
        &self,
        renderer: &mut GlesRenderer,
        icon: CursorIcon,
        scale: i32,
        cursor: &XCursor,
        idx: usize,
    ) -> GlesTexture {
        let context_id = renderer.context_id();
        self.coverage
            .borrow_mut()
            .entry((icon, scale, idx))
            .and_modify(|(cached_context, texture)| {
                if *cached_context != context_id {
                    let frame = &cursor.frames()[idx];
                    *texture = import_coverage(renderer, frame);
                    *cached_context = context_id.clone();
                }
            })
            .or_insert_with(|| {
                let frame = &cursor.frames()[idx];
                (context_id, import_coverage(renderer, frame))
            })
            .1
            .clone()
    }

    /// Get a cacheable signed-distance raster for a named cursor frame. Distances
    /// are negative inside the alpha silhouette and positive outside, in pixels.
    pub fn get_sdf(
        &self,
        renderer: &mut GlesRenderer,
        icon: CursorIcon,
        scale: i32,
        cursor: &XCursor,
        idx: usize,
    ) -> GlesTexture {
        let context_id = renderer.context_id();
        self.sdf
            .borrow_mut()
            .entry((icon, scale, idx))
            .and_modify(|(cached_context, texture)| {
                if *cached_context != context_id {
                    *texture = import_sdf(renderer, &cursor.frames()[idx]);
                    *cached_context = context_id.clone();
                }
            })
            .or_insert_with(|| (context_id, import_sdf(renderer, &cursor.frames()[idx])))
            .1
            .clone()
    }
}

fn import_sdf(renderer: &mut GlesRenderer, frame: &Image) -> GlesTexture {
    let size = (frame.width as i32, frame.height as i32).into();
    let pixels = signed_distance_raster(
        &frame.pixels_rgba,
        frame.width as usize,
        frame.height as usize,
    );
    let texture = renderer
        .with_context(|gl| unsafe {
            let mut tex = 0;
            gl.GenTextures(1, &mut tex);
            gl.BindTexture(ffi::TEXTURE_2D, tex);
            gl.TexImage2D(
                ffi::TEXTURE_2D,
                0,
                ffi::R32F as i32,
                frame.width as i32,
                frame.height as i32,
                0,
                ffi::RED,
                ffi::FLOAT,
                pixels.as_ptr().cast(),
            );
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
            gl.TexParameteri(
                ffi::TEXTURE_2D,
                ffi::TEXTURE_WRAP_S,
                ffi::CLAMP_TO_EDGE as i32,
            );
            gl.TexParameteri(
                ffi::TEXTURE_2D,
                ffi::TEXTURE_WRAP_T,
                ffi::CLAMP_TO_EDGE as i32,
            );
            tex
        })
        .expect("error importing cursor SDF texture");
    unsafe { GlesTexture::from_raw(renderer, Some(ffi::R32F), false, texture, size) }
}

/// Build a signed Euclidean-distance field from premultiplied ARGB pixels.
/// The input is top-left origin; negative values are inside the alpha silhouette.
fn signed_distance_raster(pixels: &[u8], width: usize, height: usize) -> Vec<f32> {
    let inside: Vec<_> = pixels.chunks_exact(4).map(|p| p[3] >= 128).collect();
    let mut result = vec![0.0; inside.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let want_inside = inside[i];
            let mut nearest = f32::INFINITY;
            for yy in 0..height {
                for xx in 0..width {
                    if inside[yy * width + xx] != want_inside {
                        let dx = x as f32 - xx as f32;
                        let dy = y as f32 - yy as f32;
                        nearest = nearest.min((dx * dx + dy * dy).sqrt());
                    }
                }
            }
            result[i] = if want_inside { -nearest } else { nearest };
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::signed_distance_raster;

    #[test]
    fn signed_distance_is_negative_inside_and_positive_outside() {
        let pixels = [0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0];
        let sdf = signed_distance_raster(&pixels, 3, 1);
        assert!(sdf[0] > 0.0);
        assert!(sdf[1] < 0.0);
        assert!(sdf[2] > 0.0);
    }
}

fn import_coverage(renderer: &mut GlesRenderer, frame: &Image) -> GlesTexture {
    renderer
        .import_memory(
            &frame.pixels_rgba,
            Fourcc::Argb8888,
            (frame.width as i32, frame.height as i32).into(),
            true,
        )
        .expect("error importing cursor coverage texture")
}

// The XCursorBuffer implementation is inspired by `wayland-rs`, thus provided under MIT license.

/// The state of the `NamedCursor`.
pub struct XCursor {
    /// The image for the underlying named cursor.
    images: Vec<Image>,
    /// The total duration of the animation.
    animation_duration: u32,
}

impl XCursor {
    /// Given a time, calculate which frame to show, and how much time remains until the next frame.
    ///
    /// Time will wrap, so if for instance the cursor has an animation lasting 100ms,
    /// then calling this function with 5ms and 105ms as input gives the same output.
    pub fn frame(&self, mut millis: u32) -> (usize, &Image) {
        if self.animation_duration == 0 {
            return (0, &self.images[0]);
        }

        millis %= self.animation_duration;

        let mut res = 0;
        for (i, img) in self.images.iter().enumerate() {
            if millis < img.delay {
                res = i;
                break;
            }
            millis -= img.delay;
        }

        (res, &self.images[res])
    }

    /// Get the frames for the given `XCursor`.
    pub fn frames(&self) -> &[Image] {
        &self.images
    }

    /// Check whether the cursor is animated.
    pub fn is_animated_cursor(&self) -> bool {
        self.images.len() > 1
    }

    /// Get hotspot for the given `image`.
    pub fn hotspot(image: &Image) -> Point<i32, Physical> {
        (image.xhot as i32, image.yhot as i32).into()
    }
}

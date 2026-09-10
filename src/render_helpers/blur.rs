use std::cmp::{max, min};
use std::iter::{once, zip};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{ensure, Context as _};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::gles::{ffi, link_program, GlesError, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{ContextId, Offscreen as _, Renderer as _, Texture as _};
use smithay::gpu_span_location;
use smithay::utils::{Buffer, Size};

use crate::render_helpers::custom_blur::{MaskPassStep, PipelineConfig, RenderPassStep};
use crate::render_helpers::shaders::Shaders;

#[derive(Debug)]
pub struct Blur {
    program: BlurProgram,
    /// Context ID of the renderer that created the program and the textures.
    renderer_context_id: ContextId<GlesTexture>,
    /// Output texture followed by intermediate textures, large to small.
    ///
    /// Created lazily and stored here to avoid recreating blur textures frequently.
    textures: Vec<GlesTexture>,
    /// Intermediate textures for custom blur passes.
    custom_textures: Vec<GlesTexture>,
    /// Final custom-pipeline output multiplied by the exact region mask.
    masked_output_texture: Option<GlesTexture>,
    /// Mask texture at source resolution, rendered from GPU mask shader.
    /// Primary buffer in the mask ping-pong pair.
    mask_texture_a: Option<GlesTexture>,
    /// Secondary buffer for mask ping-pong.
    mask_texture_b: Option<GlesTexture>,
    /// Compiled analytical full-window mask program.
    mask_program: Option<MaskProgram>,
    /// Program applying the exact mask to a custom-pipeline output.
    mask_output_program: Option<MaskOutputProgram>,
    /// Cached mask inputs and final texture. A matching custom pipeline and
    /// geometry can reuse the fully rendered mask without touching either
    /// ping-pong texture.
    cached_mask_rects: Arc<Vec<[f32; 4]>>,
    cached_mask_w: i32,
    cached_mask_h: i32,
    cached_mask_geo_size: (f32, f32),
    cached_mask_corner_radius: [f32; 4],
    cached_mask_pipeline: Option<Rc<CustomBlurProgramInner>>,
    cached_mask_texture_is_b: Option<bool>,
    /// Source-pixel rectangle data shared by all mask paths.
    cached_rects_px: Vec<[f32; 4]>,
    jfa_pipeline: Option<JfaPipeline>,
    jfa_textures: Option<JfaTextures>,
    /// Compiled instanced binary-mask program. Used by the source-sized
    /// binary mask and the bbox-local JFA input.
    binary_program: Option<JfaBinaryProgram>,
    /// 1×N RGBA32F texture holding the subregion rects (one texel per
    /// rect, RGBA = x1,y1,x2,y2 in source pixels). Grown when the rect count
    /// exceeds capacity; never shrunk.
    rects_texture: Option<GlesTexture>,
    rects_capacity: i32,
    /// Cache for the JFA mask pipeline. The whole pipeline is computed in
    /// bbox-local coordinates, so its output (`jfa_textures.encoded`) is
    /// invariant under cursor motion — only the absolute bbox origin
    /// changes. When these values match the previous frame, we skip the
    /// entire pipeline and just re-blit the cached encoded texture at
    /// the new screen position.
    cached_jfa_rects_local: Vec<[f32; 4]>,
    cached_jfa_bbox_size: Option<(i32, i32)>,
    cached_jfa_bbox_origin: Option<(i32, i32)>,
    /// Full-source mask texture that received the cached bbox-local JFA output.
    cached_jfa_mask_texture_id: Option<u32>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct BlurOptions {
    pub passes: u8,
    pub offset: f64,
    pub geo_size: (f32, f32),
    pub corner_radius: [f32; 4],
    pub subregion_rects: Arc<Vec<[f32; 4]>>,
    pub window_screen_rect: [f32; 4],
    /// Inline custom shader pipeline definition, or `None` to use the
    /// default Kawase blur. Cached in `Shaders` keyed by a hash of the
    /// shader source strings.
    pub shader_pipeline: Option<niri_config::ShaderPipeline>,
}

impl BlurOptions {
    pub fn clamped_passes(&self) -> usize {
        self.passes.clamp(1, MAX_KAWASE_PASSES) as usize
    }

    pub fn with_geometry(mut self, geo_size: (f32, f32), corner_radius: [f32; 4]) -> Self {
        self.geo_size = geo_size;
        self.corner_radius = corner_radius;
        self
    }

    pub fn with_subregion_rects(mut self, rects: Arc<Vec<[f32; 4]>>) -> Self {
        self.subregion_rects = rects;
        self
    }

    pub fn with_window_screen_rect(mut self, rect: [f32; 4]) -> Self {
        self.window_screen_rect = rect;
        self
    }
}

impl From<niri_config::Blur> for BlurOptions {
    fn from(config: niri_config::Blur) -> Self {
        Self {
            passes: config.passes,
            offset: config.offset,
            geo_size: (0.0, 0.0),
            corner_radius: [0.0; 4],
            subregion_rects: Arc::new(Vec::new()),
            window_screen_rect: [0.0; 4],
            shader_pipeline: config.shader_pipeline,
        }
    }
}

#[derive(Debug)]
pub struct BlurOutput {
    pub texture: GlesTexture,
    /// The texture is transparent outside the requested protocol subregion.
    pub subregion_clipped: bool,
}

#[derive(Debug, Clone)]
pub struct BlurProgram(Rc<BlurProgramInner>);

#[derive(Debug)]
struct BlurProgramInner {
    down: BlurProgramInternal,
    up: BlurProgramInternal,
}

#[derive(Debug)]
struct BlurProgramInternal {
    program: ffi::types::GLuint,
    uniform_tex: ffi::types::GLint,
    uniform_half_pixel: ffi::types::GLint,
    uniform_offset: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

unsafe fn compile_program(gl: &ffi::Gles2, src: &str) -> Result<BlurProgramInternal, GlesError> {
    let program = unsafe { link_program(gl, include_str!("shaders/blur.vert"), src)? };

    let vert = c"vert";
    let tex = c"tex";
    let half_pixel = c"half_pixel";
    let offset = c"offset";

    Ok(BlurProgramInternal {
        program,
        uniform_tex: gl.GetUniformLocation(program, tex.as_ptr()),
        uniform_half_pixel: gl.GetUniformLocation(program, half_pixel.as_ptr()),
        uniform_offset: gl.GetUniformLocation(program, offset.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}

impl BlurProgram {
    pub fn compile(renderer: &mut GlesRenderer) -> anyhow::Result<Self> {
        renderer
            .with_context(move |gl| unsafe {
                let down = compile_program(gl, include_str!("shaders/blur_down.frag"))
                    .context("error compiling blur_down shader")?;
                let up = compile_program(gl, include_str!("shaders/blur_up.frag"))
                    .context("error compiling blur_up shader")?;
                Ok(Self(Rc::new(BlurProgramInner { down, up })))
            })
            .context("error making GL context current")?
    }

    pub fn destroy(self, renderer: &mut GlesRenderer) -> Result<(), GlesError> {
        renderer.with_context(move |gl| unsafe {
            gl.DeleteProgram(self.0.down.program);
            gl.DeleteProgram(self.0.up.program);
        })
    }
}

#[derive(Debug)]
struct CustomBlurPassProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_input_size: ffi::types::GLint,
    uniform_half_pixel: ffi::types::GLint,
    uniform_pass: ffi::types::GLint,
    uniform_pass_count: ffi::types::GLint,
    uniform_geo_size: ffi::types::GLint,
    uniform_corner_radius: ffi::types::GLint,
    uniform_mask: ffi::types::GLint,
    uniform_window_screen_rect: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct CustomBlurProgramInner {
    mask_passes: Vec<MaskPassStep>,
    mask_programs: Vec<Option<CustomMaskProgram>>,
    render_passes: Vec<RenderPassStep>,
    render_programs: Vec<Option<CustomBlurPassProgram>>,
}

#[derive(Debug, Clone)]
pub struct CustomBlurProgram(Rc<CustomBlurProgramInner>);

unsafe fn compile_custom_pass(
    gl: &ffi::Gles2,
    frag_src: &str,
) -> Result<CustomBlurPassProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };

    let input = c"niri_input";
    let output_size = c"niri_output_size";
    let input_size = c"niri_input_size";
    let half_pixel = c"niri_half_pixel";
    let pass = c"niri_pass";
    let pass_count = c"niri_pass_count";
    let geo_size = c"niri_geo_size";
    let corner_radius = c"niri_corner_radius";
    let mask = c"niri_mask";
    let window_screen_rect = c"niri_window_screen_rect";
    let vert = c"vert";

    Ok(CustomBlurPassProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, input.as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, output_size.as_ptr()),
        uniform_input_size: gl.GetUniformLocation(program, input_size.as_ptr()),
        uniform_half_pixel: gl.GetUniformLocation(program, half_pixel.as_ptr()),
        uniform_pass: gl.GetUniformLocation(program, pass.as_ptr()),
        uniform_pass_count: gl.GetUniformLocation(program, pass_count.as_ptr()),
        uniform_geo_size: gl.GetUniformLocation(program, geo_size.as_ptr()),
        uniform_corner_radius: gl.GetUniformLocation(program, corner_radius.as_ptr()),
        uniform_mask: gl.GetUniformLocation(program, mask.as_ptr()),
        uniform_window_screen_rect: gl.GetUniformLocation(program, window_screen_rect.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}

unsafe fn compile_mask_output_program(gl: &ffi::Gles2) -> Result<MaskOutputProgram, GlesError> {
    let program = unsafe {
        link_program(
            gl,
            include_str!("shaders/blur_custom.vert"),
            include_str!("shaders/mask_output.frag"),
        )?
    };
    Ok(MaskOutputProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_mask: gl.GetUniformLocation(program, c"niri_mask".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_custom_mask_pass(
    gl: &ffi::Gles2,
    frag_src: &str,
) -> Result<CustomMaskProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };

    let subregion_count = c"niri_subregion_count";
    let subregion_rects = c"niri_subregion_rects";
    let output_size = c"niri_output_size";
    let bbox_origin = c"niri_bbox_origin";
    let geo_size = c"niri_geo_size";
    let corner_radius = c"niri_corner_radius";
    let vert = c"vert";

    Ok(CustomMaskProgram {
        program,
        uniform_subregion_count: gl.GetUniformLocation(program, subregion_count.as_ptr()),
        uniform_subregion_rects: gl.GetUniformLocation(program, subregion_rects.as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, output_size.as_ptr()),
        uniform_bbox_origin: gl.GetUniformLocation(program, bbox_origin.as_ptr()),
        uniform_geo_size: gl.GetUniformLocation(program, geo_size.as_ptr()),
        uniform_corner_radius: gl.GetUniformLocation(program, corner_radius.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}

impl CustomBlurProgram {
    pub fn compile(renderer: &mut GlesRenderer, config: &PipelineConfig) -> anyhow::Result<Self> {
        renderer
            .with_context(move |gl| unsafe {
                let mut mask_programs = Vec::with_capacity(config.mask_passes.len());
                for (i, step) in config.mask_passes.iter().enumerate() {
                    let prog = match step {
                        MaskPassStep::WindowVectors | MaskPassStep::RegionVectors => None,
                        MaskPassStep::Custom { source, name, .. } => {
                            Some(compile_custom_mask_pass(gl, source).with_context(|| {
                                format!("error compiling custom mask pass {} ({:?})", i, name)
                            })?)
                        }
                    };
                    mask_programs.push(prog);
                }

                let mut render_programs = Vec::with_capacity(config.render_passes.len());
                for (i, step) in config.render_passes.iter().enumerate() {
                    let prog = match step {
                        RenderPassStep::DualKawaseBlur { .. } => None,
                        RenderPassStep::Custom { source, name, .. } => {
                            Some(compile_custom_pass(gl, source).with_context(|| {
                                format!("error compiling custom render pass {} ({:?})", i, name)
                            })?)
                        }
                    };
                    render_programs.push(prog);
                }
                Ok(Self(Rc::new(CustomBlurProgramInner {
                    mask_passes: config.mask_passes.clone(),
                    mask_programs,
                    render_passes: config.render_passes.clone(),
                    render_programs,
                })))
            })
            .context("error making GL context current")?
    }

    pub fn destroy(self, renderer: &mut GlesRenderer) -> Result<(), GlesError> {
        renderer.with_context(move |gl| unsafe {
            for prog in &self.0.mask_programs {
                if let Some(p) = prog {
                    gl.DeleteProgram(p.program);
                }
            }
            for prog in &self.0.render_programs {
                if let Some(p) = prog {
                    gl.DeleteProgram(p.program);
                }
            }
        })
    }
}

#[derive(Debug)]
struct MaskProgram {
    program: ffi::types::GLuint,
    uniform_geo_size: ffi::types::GLint,
    uniform_corner_radius: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct MaskOutputProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_mask: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct CustomMaskProgram {
    program: ffi::types::GLuint,
    uniform_subregion_count: ffi::types::GLint,
    uniform_subregion_rects: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_bbox_origin: ffi::types::GLint,
    uniform_geo_size: ffi::types::GLint,
    uniform_corner_radius: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaBinaryProgram {
    program: ffi::types::GLuint,
    uniform_subregion_rects: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_bbox_origin: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaInitProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaStepProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_step: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonInitRhsProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_f_scale: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonRestrictMaskProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_src_texel: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonJacobiProgram {
    program: ffi::types::GLuint,
    uniform_u_in: ffi::types::GLint,
    uniform_rhs: ffi::types::GLint,
    uniform_texel: ffi::types::GLint,
    uniform_h_sq: ffi::types::GLint,
    uniform_omega: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonResidualRestrictProgram {
    program: ffi::types::GLuint,
    uniform_u_fine: ffi::types::GLint,
    uniform_rhs_fine: ffi::types::GLint,
    uniform_fine_texel: ffi::types::GLint,
    uniform_h_sq_fine: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPoissonProlongateProgram {
    program: ffi::types::GLuint,
    uniform_u_fine: ffi::types::GLint,
    uniform_correction: ffi::types::GLint,
    uniform_rhs_fine: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaEncodeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_poisson_u: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaSmoothProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPipeline {
    binary_prog: JfaBinaryProgram,
    init_prog: JfaInitProgram,
    step_prog: JfaStepProgram,
    poisson_init_rhs_prog: JfaPoissonInitRhsProgram,
    poisson_restrict_mask_prog: JfaPoissonRestrictMaskProgram,
    poisson_jacobi_prog: JfaPoissonJacobiProgram,
    poisson_residual_restrict_prog: JfaPoissonResidualRestrictProgram,
    poisson_prolongate_prog: JfaPoissonProlongateProgram,
    encode_prog: JfaEncodeProgram,
    smooth_prog: JfaSmoothProgram,
}

#[derive(Debug)]
struct MultigridLevel {
    /// Ping-pong A for the solution `u` (R channel).
    u_a: GlesTexture,
    /// Ping-pong B for the solution `u` (R channel).
    u_b: GlesTexture,
    /// Packed RHS: R = `f` (forcing function), G = mask.
    rhs: GlesTexture,
    /// Level size.
    size: Size<i32, Buffer>,
}
#[derive(Debug)]
struct JfaTextures {
    /// Binary mask of the subregion union (R16F).
    bin: GlesTexture,
    /// JFA ping-pong A (RGBA16F, nearest exterior coordinate).
    jfa_a: GlesTexture,
    /// JFA ping-pong B (RGBA16F, nearest exterior coordinate).
    jfa_b: GlesTexture,
    /// Multigrid pyramid. Index 0 is the finest (bbox-sized) level.
    pyramid: Vec<MultigridLevel>,
    /// Final encoded output (RGBA16F: R=mask, GB=direction).
    encoded: GlesTexture,
    /// Bbox size these textures were allocated for.
    size: Size<i32, Buffer>,
}

const FULLSCREEN_TRI_STRIP: [f32; 12] =
    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

/// Maximum number of Kawase blur down+up passes.
const MAX_KAWASE_PASSES: u8 = 31;

/// Per-component comparison epsilon for the JFA bbox-local rects cache.
const JFA_CACHE_EPSILON: f32 = 0.01;

/// Maximum number of multigrid levels in the Poisson pyramid. The pyramid
/// stops early once both dimensions reach a genuinely small coarse grid.
const MAX_MULTIGRID_LEVELS: usize = 12;

/// Coarse-grid target for the Poisson solve.
const MULTIGRID_COARSE_SIZE: i32 = 8;

/// Weighted-Jacobi damping factor for the Poisson smoother.
const JACOBI_OMEGA: f32 = 0.8;

/// Number of Jacobi smoothing sweeps on each side of prolongation.
const JACOBI_SWEEPS: i32 = 3;

/// Number of V-cycles per Poisson solve.
const V_CYCLES: usize = 1;

/// Pixels of exterior border padding around the JFA bbox.
const BBOX_BORDER: i32 = 1;

unsafe fn check_gl_error(gl: &ffi::Gles2) {
    let mut had_error = false;
    loop {
        let err = gl.GetError();
        if err == ffi::NO_ERROR {
            break;
        }
        had_error = true;
        warn!("GL error: 0x{:x}", err);
    }
    if had_error {
        warn!("GL errors detected during rendering pass");
    }
}

unsafe fn draw_fullscreen_quad(gl: &ffi::Gles2, attrib_vert: ffi::types::GLint) {
    gl.EnableVertexAttribArray(attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        FULLSCREEN_TRI_STRIP.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(attrib_vert as u32);
}

// Allocates a renderable floating-point GlesTexture directly via raw GL,
// bypassing smithay's Fourcc mapping. The repository already requires the
// float color-buffer and linear-filter extensions for the JFA pipeline.
fn create_float_buffer(
    renderer: &mut GlesRenderer,
    size: Size<i32, Buffer>,
    internal_format: ffi::types::GLenum,
    format: ffi::types::GLenum,
) -> Result<GlesTexture, GlesError> {
    let tex = renderer.with_context(|gl| unsafe {
        let mut tex = 0;
        gl.GenTextures(1, &mut tex);
        gl.BindTexture(ffi::TEXTURE_2D, tex);
        gl.TexImage2D(
            ffi::TEXTURE_2D,
            0,
            internal_format as i32,
            size.w,
            size.h,
            0,
            format,
            ffi::FLOAT,
            std::ptr::null(),
        );
        gl.TexParameteri(
            ffi::TEXTURE_2D,
            ffi::TEXTURE_MIN_FILTER,
            ffi::NEAREST as i32,
        );
        gl.TexParameteri(
            ffi::TEXTURE_2D,
            ffi::TEXTURE_MAG_FILTER,
            ffi::NEAREST as i32,
        );
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
    })?;
    Ok(unsafe { GlesTexture::from_raw(renderer, Some(internal_format), false, tex, size) })
}

unsafe fn compile_mask_program(gl: &ffi::Gles2) -> Result<MaskProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/mask.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };

    let geo_size = c"niri_geo_size";
    let corner_radius = c"niri_corner_radius";
    let vert = c"vert";

    Ok(MaskProgram {
        program,
        uniform_geo_size: gl.GetUniformLocation(program, geo_size.as_ptr()),
        uniform_corner_radius: gl.GetUniformLocation(program, corner_radius.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}

unsafe fn compile_binary_program(gl: &ffi::Gles2) -> Result<JfaBinaryProgram, GlesError> {
    let vert_src = include_str!("shaders/mask_binary.vert");
    let frag_src = include_str!("shaders/mask_binary.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaBinaryProgram {
        program,
        uniform_subregion_rects: gl.GetUniformLocation(program, c"niri_subregion_rects".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_bbox_origin: gl.GetUniformLocation(program, c"niri_bbox_origin".as_ptr()),
    })
}

unsafe fn compile_jfa_init(gl: &ffi::Gles2) -> Result<JfaInitProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_init.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaInitProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_step(gl: &ffi::Gles2) -> Result<JfaStepProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_step.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaStepProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_step: gl.GetUniformLocation(program, c"niri_step".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_init_rhs(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonInitRhsProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_init_rhs.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonInitRhsProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_f_scale: gl.GetUniformLocation(program, c"niri_f_scale".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_restrict_mask(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonRestrictMaskProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_restrict_mask.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonRestrictMaskProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_src_texel: gl.GetUniformLocation(program, c"niri_src_texel".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_jacobi(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonJacobiProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_jacobi.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonJacobiProgram {
        program,
        uniform_u_in: gl.GetUniformLocation(program, c"niri_u_in".as_ptr()),
        uniform_rhs: gl.GetUniformLocation(program, c"niri_rhs".as_ptr()),
        uniform_texel: gl.GetUniformLocation(program, c"niri_texel".as_ptr()),
        uniform_h_sq: gl.GetUniformLocation(program, c"niri_h_sq".as_ptr()),
        uniform_omega: gl.GetUniformLocation(program, c"niri_omega".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_residual_restrict(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonResidualRestrictProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_residual_restrict.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonResidualRestrictProgram {
        program,
        uniform_u_fine: gl.GetUniformLocation(program, c"niri_u_fine".as_ptr()),
        uniform_rhs_fine: gl.GetUniformLocation(program, c"niri_rhs_fine".as_ptr()),
        uniform_fine_texel: gl.GetUniformLocation(program, c"niri_fine_texel".as_ptr()),
        uniform_h_sq_fine: gl.GetUniformLocation(program, c"niri_h_sq_fine".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_poisson_prolongate(
    gl: &ffi::Gles2,
) -> Result<JfaPoissonProlongateProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_poisson_prolongate.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaPoissonProlongateProgram {
        program,
        uniform_u_fine: gl.GetUniformLocation(program, c"niri_u_fine".as_ptr()),
        uniform_correction: gl.GetUniformLocation(program, c"niri_correction".as_ptr()),
        uniform_rhs_fine: gl.GetUniformLocation(program, c"niri_rhs_fine".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_encode(gl: &ffi::Gles2) -> Result<JfaEncodeProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_encode.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaEncodeProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_poisson_u: gl.GetUniformLocation(program, c"niri_poisson_u".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_max_dist: gl.GetUniformLocation(program, c"niri_max_dist".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

unsafe fn compile_jfa_smooth(gl: &ffi::Gles2) -> Result<JfaSmoothProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_smooth.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaSmoothProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
    })
}

impl Blur {
    pub fn new(renderer: &mut GlesRenderer) -> Option<Self> {
        let program = Shaders::get(renderer).blur.clone()?;
        Some(Self {
            program,
            renderer_context_id: renderer.context_id(),
            textures: Vec::new(),
            custom_textures: Vec::new(),
            mask_texture_a: None,
            mask_texture_b: None,
            mask_program: None,
            masked_output_texture: None,
            cached_mask_rects: Arc::new(Vec::new()),
            cached_mask_w: 0,
            cached_mask_h: 0,
            mask_output_program: None,
            cached_mask_geo_size: (0.0, 0.0),
            cached_mask_corner_radius: [0.0; 4],
            cached_mask_pipeline: None,
            cached_mask_texture_is_b: None,
            cached_rects_px: Vec::new(),
            jfa_pipeline: None,
            jfa_textures: None,
            binary_program: None,
            rects_texture: None,
            rects_capacity: 0,
            cached_jfa_rects_local: Vec::new(),
            cached_jfa_bbox_size: None,
            cached_jfa_bbox_origin: None,
            cached_jfa_mask_texture_id: None,
        })
    }

    pub fn context_id(&self) -> ContextId<GlesTexture> {
        self.renderer_context_id.clone()
    }

    pub fn prepare_textures(
        &mut self,
        create_texture: impl FnMut(Fourcc, Size<i32, Buffer>) -> Result<GlesTexture, GlesError>,
        source: &GlesTexture,
        options: &BlurOptions,
    ) -> anyhow::Result<()> {
        let _span = tracy_client::span!("Blur::prepare_textures");

        // Custom pipelines manage their own textures in render_custom().
        // Don't allocate the Kawase cascade when a shader pipeline is set.
        if options.shader_pipeline.is_some() {
            self.textures.clear();
            return Ok(());
        }

        let passes = options.clamped_passes();
        self.ensure_kawase_textures(create_texture, source, passes)
    }

    fn ensure_kawase_textures(
        &mut self,
        mut create_texture: impl FnMut(Fourcc, Size<i32, Buffer>) -> Result<GlesTexture, GlesError>,
        source: &GlesTexture,
        passes: usize,
    ) -> anyhow::Result<()> {
        let size = source.size();

        if let Some(output) = self.textures.first_mut() {
            let old_size = output.size();
            if old_size != size {
                trace!(
                    "recreating textures: output size changed from {} × {} to {} × {}",
                    old_size.w,
                    old_size.h,
                    size.w,
                    size.h
                );
                self.textures.clear();
            } else if !output.is_unique_reference() {
                debug!("recreating textures: not unique",);
                // We only need to recreate the output texture here, but this case shouldn't really
                // happen anyway, and this is simpler.
                self.textures.clear();
            }
        }

        // Create any missing textures.
        let mut w = size.w;
        let mut h = size.h;
        for i in 0..=passes {
            let size = Size::new(w, h);
            w = max(1, w / 2);
            h = max(1, h / 2);

            if self.textures.len() > i {
                // This texture already exists.
                continue;
            }

            // debug!("creating texture for step {i} sized {w} × {h}");

            let texture: GlesTexture =
                create_texture(Fourcc::Abgr8888, size).context("error creating texture")?;
            self.textures.push(texture);
        }

        // Drop any no longer needed textures.
        self.textures.drain(passes + 1..);

        Ok(())
    }

    fn can_gpu_clip_custom_output(mask_passes: &[MaskPassStep], has_subregions: bool) -> bool {
        has_subregions && matches!(mask_passes.last(), None | Some(MaskPassStep::RegionVectors))
    }

    fn render_custom(
        &mut self,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        custom_program: &CustomBlurProgram,
        options: &BlurOptions,
    ) -> anyhow::Result<BlurOutput> {
        let _span = tracy_client::span!("Blur::render_custom");
        trace!("rendering custom blur");

        let inner = &custom_program.0;
        let geo_size = options.geo_size;
        let corner_radius = options.corner_radius;

        ensure!(
            renderer.context_id() == self.renderer_context_id,
            "wrong renderer"
        );

        let render_passes = &inner.render_passes;
        let render_programs = &inner.render_programs;
        let pass_count = render_passes.len();

        let source_size = source.size();
        let source_w = source_size.w as f32;
        let source_h = source_size.h as f32;

        let mut pass_output_sizes: Vec<(i32, i32)> = Vec::with_capacity(pass_count);
        let mut current_w = source_w;
        let mut current_h = source_h;
        for step in render_passes {
            let scale = match step {
                RenderPassStep::DualKawaseBlur { .. } => 1.0,
                RenderPassStep::Custom { scale, .. } => *scale,
            };
            current_w = (current_w * scale).max(1.0);
            current_h = (current_h * scale).max(1.0);
            pass_output_sizes.push((current_w as i32, current_h as i32));
        }
        let (final_output_w, final_output_h) = pass_output_sizes
            .last()
            .copied()
            .context("custom pipeline has no render passes")?;
        let final_output_size = Size::new(final_output_w, final_output_h);
        let mut apply_output_mask = Self::can_gpu_clip_custom_output(
            &inner.mask_passes,
            !options.subregion_rects.is_empty(),
        );

        let needs_recreate = self.custom_textures.len() != pass_count
            || self
                .custom_textures
                .iter()
                .zip(pass_output_sizes.iter())
                .any(|(tex, &size)| tex.size().w != size.0 || tex.size().h != size.1);

        if needs_recreate {
            self.custom_textures.clear();
            for &(w, h) in &pass_output_sizes {
                let size = Size::new(w, h);
                let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr8888, size)?;
                self.custom_textures.push(texture);
            }
        }

        if apply_output_mask
            && self
                .masked_output_texture
                .as_ref()
                .is_none_or(|texture| texture.size() != final_output_size)
        {
            self.masked_output_texture =
                Some(renderer.create_buffer(Fourcc::Abgr8888, final_output_size)?);
        }

        let mask_w = source_size.w;
        let mask_h = source_size.h;

        let rects_changed = !Arc::ptr_eq(&self.cached_mask_rects, &options.subregion_rects);
        let size_changed = self.cached_mask_w != mask_w || self.cached_mask_h != mask_h;
        let mask_geometry_changed = self.cached_mask_geo_size != geo_size
            || self.cached_mask_corner_radius != corner_radius;
        let pipeline_changed = self
            .cached_mask_pipeline
            .as_ref()
            .is_none_or(|cached| !Rc::ptr_eq(cached, &custom_program.0));
        let mask_inputs_changed =
            rects_changed || size_changed || mask_geometry_changed || pipeline_changed;

        if rects_changed
            || size_changed
            || self.cached_rects_px.len() != options.subregion_rects.len()
        {
            self.cached_rects_px = options
                .subregion_rects
                .iter()
                .map(|r| {
                    [
                        r[0] * source_size.w as f32,
                        r[1] * source_size.h as f32,
                        r[2] * source_size.w as f32,
                        r[3] * source_size.h as f32,
                    ]
                })
                .collect();
        }

        let mask_size = Size::new(mask_w, mask_h);
        let need_new_mask = self
            .mask_texture_a
            .as_ref()
            .is_none_or(|t| t.size() != mask_size);
        if need_new_mask {
            let tex: GlesTexture = renderer.create_buffer(Fourcc::Abgr16161616f, mask_size)?;
            self.mask_texture_a = Some(tex);
        }
        let need_mask_b = self
            .mask_texture_b
            .as_ref()
            .is_none_or(|t| t.size() != mask_size);
        if need_mask_b {
            let tex: GlesTexture = renderer.create_buffer(Fourcc::Abgr16161616f, mask_size)?;
            self.mask_texture_b = Some(tex);
        }

        if need_new_mask || need_mask_b {
            self.cached_mask_texture_is_b = None;
        }

        let needed = options.subregion_rects.len() as i32;
        let mut rects_texture_changed = false;
        if needed > self.rects_capacity {
            let capacity = ((needed as u32).next_power_of_two() as i32).max(16);
            let size = Size::new(capacity, 1);
            self.rects_texture = Some(create_float_buffer(
                renderer,
                size,
                ffi::RGBA32F,
                ffi::RGBA,
            )?);
            self.rects_capacity = capacity;
            rects_texture_changed = true;
        }

        let has_region_vectors = inner
            .mask_passes
            .iter()
            .any(|s| matches!(s, MaskPassStep::RegionVectors));
        let mut jfa_need_alloc = false;
        let jfa_bbox = if has_region_vectors && !options.subregion_rects.is_empty() {
            let mut bbox = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
            for rect in options.subregion_rects.iter() {
                bbox[0] = bbox[0].min(rect[0]);
                bbox[1] = bbox[1].min(rect[1]);
                bbox[2] = bbox[2].max(rect[2]);
                bbox[3] = bbox[3].max(rect[3]);
            }
            const EPS: f32 = 1e-4;
            let bbx = ((bbox[0] * source_size.w as f32 + EPS).floor() as i32 - BBOX_BORDER).max(0);
            let bby = ((bbox[1] * source_size.h as f32 + EPS).floor() as i32 - BBOX_BORDER).max(0);
            let bbw = (((bbox[2] - bbox[0]) * source_size.w as f32 - EPS).ceil() as i32
                + 2 * BBOX_BORDER)
                .max(1)
                .min(source_size.w - bbx);
            let bbh = (((bbox[3] - bbox[1]) * source_size.h as f32 - EPS).ceil() as i32
                + 2 * BBOX_BORDER)
                .max(1)
                .min(source_size.h - bby);
            if bbw > 0 && bbh > 0 {
                let bbox_size = Size::new(bbw, bbh);
                let need_alloc = match &self.jfa_textures {
                    Some(t) => t.size != bbox_size,
                    None => true,
                };
                jfa_need_alloc = need_alloc;
                if need_alloc {
                    let mut pyramid: Vec<MultigridLevel> = Vec::with_capacity(MAX_MULTIGRID_LEVELS);
                    for k in 0..MAX_MULTIGRID_LEVELS {
                        let mw = std::cmp::max(1, bbw >> k);
                        let mh = std::cmp::max(1, bbh >> k);
                        let lvl_size = Size::new(mw, mh);
                        pyramid.push(MultigridLevel {
                            u_a: create_float_buffer(renderer, lvl_size, ffi::R32F, ffi::RED)?,
                            u_b: create_float_buffer(renderer, lvl_size, ffi::R32F, ffi::RED)?,
                            rhs: create_float_buffer(renderer, lvl_size, ffi::RG16F, ffi::RG)?,
                            size: lvl_size,
                        });
                        if mw <= MULTIGRID_COARSE_SIZE && mh <= MULTIGRID_COARSE_SIZE {
                            break;
                        }
                    }
                    self.jfa_textures = Some(JfaTextures {
                        bin: create_float_buffer(renderer, bbox_size, ffi::R16F, ffi::RED)?,
                        jfa_a: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                        jfa_b: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                        pyramid,
                        encoded: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                        size: bbox_size,
                    });
                }
                Some((bbx, bby, bbw, bbh))
            } else {
                None
            }
        } else {
            None
        };

        let mut jfa_cache_previous_bbox = None;
        let jfa_cache_hit = match jfa_bbox {
            Some((bbx, bby, bbw, bbh)) if !jfa_need_alloc => {
                let rects_local: Vec<[f32; 4]> = self
                    .cached_rects_px
                    .iter()
                    .map(|r| {
                        [
                            r[0] - bbx as f32,
                            r[1] - bby as f32,
                            r[2] - bbx as f32,
                            r[3] - bby as f32,
                        ]
                    })
                    .collect();
                let bbox_size = (bbw, bbh);
                let hit = self.cached_jfa_bbox_size == Some(bbox_size)
                    && self.cached_jfa_rects_local.len() == rects_local.len()
                    && self
                        .cached_jfa_rects_local
                        .iter()
                        .zip(rects_local.iter())
                        .all(|(a, b)| {
                            (a[0] - b[0]).abs() < JFA_CACHE_EPSILON
                                && (a[1] - b[1]).abs() < JFA_CACHE_EPSILON
                                && (a[2] - b[2]).abs() < JFA_CACHE_EPSILON
                                && (a[3] - b[3]).abs() < JFA_CACHE_EPSILON
                        });
                if hit {
                    jfa_cache_previous_bbox = self
                        .cached_jfa_bbox_origin
                        .map(|(old_x, old_y)| (old_x, old_y, bbw, bbh));
                }
                self.cached_jfa_rects_local = rects_local;
                self.cached_jfa_bbox_size = Some(bbox_size);
                self.cached_jfa_bbox_origin = Some((bbx, bby));
                hit
            }
            _ => {
                self.cached_jfa_rects_local.clear();
                self.cached_jfa_bbox_size = None;
                self.cached_jfa_bbox_origin = None;
                false
            }
        };

        let mask_passes = &inner.mask_passes;
        let mask_programs = &inner.mask_programs;

        // If any render step is DualKawaseBlur, prepare Kawase textures
        // before entering the GL context. Use the maximum passes across
        // all DualKawaseBlur steps so that intermediate textures are sized
        // for the largest chain.
        let max_kawase_passes = render_passes
            .iter()
            .filter_map(|s| match s {
                RenderPassStep::DualKawaseBlur { passes, .. } => Some(
                    passes
                        .map(|p| p.clamp(1, MAX_KAWASE_PASSES) as usize)
                        .unwrap_or_else(|| options.clamped_passes()),
                ),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if max_kawase_passes > 0 {
            self.ensure_kawase_textures(
                |fourcc, size| renderer.create_buffer(fourcc, size),
                source,
                max_kawase_passes,
            )?;
        }
        let mask_cache_hit = !need_new_mask
            && !need_mask_b
            && !mask_inputs_changed
            && self.cached_mask_texture_is_b.is_some();
        let first_mask_is_vector = matches!(
            mask_passes.first(),
            Some(MaskPassStep::WindowVectors | MaskPassStep::RegionVectors)
        );

        renderer.with_profiled_context(gpu_span_location!("Blur::render_custom"), |gl| unsafe {
            while gl.GetError() != ffi::NO_ERROR {}

            gl.Disable(ffi::BLEND);
            gl.Disable(ffi::SCISSOR_TEST);
            gl.ActiveTexture(ffi::TEXTURE0);

            let mut mask_fbo = 0u32;
            gl.GenFramebuffers(1, &mut mask_fbo);
            gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, mask_fbo);

            let rects_upload = rects_texture_changed || rects_changed || size_changed;
            if !mask_cache_hit && rects_upload {
                if let Some(rects_tex) = self.rects_texture.as_ref() {
                    if !self.cached_rects_px.is_empty() {
                        gl.ActiveTexture(ffi::TEXTURE1);
                        gl.BindTexture(ffi::TEXTURE_2D, rects_tex.tex_id());
                        gl.TexSubImage2D(
                            ffi::TEXTURE_2D,
                            0,
                            0,
                            0,
                            self.cached_rects_px.len() as i32,
                            1,
                            ffi::RGBA,
                            ffi::FLOAT,
                            self.cached_rects_px.as_ptr() as *const _,
                        );
                        gl.ActiveTexture(ffi::TEXTURE0);
                    }
                }
            }

            // Step 0: binary mask into mask_texture_a. A built-in vector
            // pass is self-contained, so it does not need this source mask.
            let mask_a = self.mask_texture_a.as_ref().unwrap();
            let mask_b = self.mask_texture_b.as_ref().unwrap();
            let mut src_mask = mask_a;
            let mut active_mask_is_b = false;

            if !mask_cache_hit {
                if !first_mask_is_vector {
                    let mask_a_id = mask_a.tex_id();
                    clear_mask_texture(gl, mask_fbo, mask_a_id);

                    if !options.subregion_rects.is_empty() {
                        if let Some(rects_tex) = self.rects_texture.as_ref() {
                            if self.binary_program.is_none() {
                                match compile_binary_program(gl) {
                                    Ok(p) => self.binary_program = Some(p),
                                    Err(err) => {
                                        warn!("error compiling binary mask shader: {err:?}");
                                    }
                                }
                            }
                            if let Some(bin_prog) = self.binary_program.as_ref() {
                                render_binary_mask(
                                    gl, bin_prog, options, mask_a, rects_tex, 0, 0, mask_w, mask_h,
                                );
                                check_gl_error(gl);
                            }
                        }
                    }

                    gl.BindTexture(ffi::TEXTURE_2D, mask_a_id);
                    gl.TexParameteri(
                        ffi::TEXTURE_2D,
                        ffi::TEXTURE_MIN_FILTER,
                        ffi::NEAREST as i32,
                    );
                    gl.TexParameteri(
                        ffi::TEXTURE_2D,
                        ffi::TEXTURE_MAG_FILTER,
                        ffi::NEAREST as i32,
                    );
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
                }

                // Step 1: mask pipeline passes (binary → mask_texture_a → passes).
                for (i, step) in mask_passes.iter().enumerate() {
                    let dst_mask = if active_mask_is_b { mask_a } else { mask_b };

                    match step {
                        MaskPassStep::WindowVectors => {
                            if self.mask_program.is_none() {
                                match compile_mask_program(gl) {
                                    Ok(p) => self.mask_program = Some(p),
                                    Err(err) => {
                                        warn!("error compiling analytical mask shader: {err:?}");
                                        src_mask = dst_mask;
                                        active_mask_is_b = !active_mask_is_b;
                                        continue;
                                    }
                                }
                            }
                            if let Some(mask_prog) = self.mask_program.as_ref() {
                                gl.FramebufferTexture2D(
                                    ffi::DRAW_FRAMEBUFFER,
                                    ffi::COLOR_ATTACHMENT0,
                                    ffi::TEXTURE_2D,
                                    dst_mask.tex_id(),
                                    0,
                                );

                                gl.UseProgram(mask_prog.program);
                                gl.Uniform2f(mask_prog.uniform_geo_size, geo_size.0, geo_size.1);
                                gl.Uniform4f(
                                    mask_prog.uniform_corner_radius,
                                    corner_radius[0],
                                    corner_radius[1],
                                    corner_radius[2],
                                    corner_radius[3],
                                );

                                gl.Viewport(0, 0, mask_w, mask_h);
                                draw_fullscreen_quad(gl, mask_prog.attrib_vert);

                                gl.BindTexture(ffi::TEXTURE_2D, dst_mask.tex_id());
                                gl.TexParameteri(
                                    ffi::TEXTURE_2D,
                                    ffi::TEXTURE_MIN_FILTER,
                                    ffi::LINEAR as i32,
                                );
                                gl.TexParameteri(
                                    ffi::TEXTURE_2D,
                                    ffi::TEXTURE_MAG_FILTER,
                                    ffi::LINEAR as i32,
                                );
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
                            }
                        }
                        MaskPassStep::RegionVectors => {
                            match (
                                jfa_bbox,
                                self.jfa_textures.as_ref(),
                                self.rects_texture.as_ref(),
                            ) {
                                (Some((bbx, bby, bbw, bbh)), Some(textures), Some(rects_tex)) => {
                                    let mask_id = dst_mask.tex_id();
                                    let previous_bbox = if i == 0
                                        && self.cached_jfa_mask_texture_id == Some(mask_id)
                                    {
                                        jfa_cache_previous_bbox
                                    } else {
                                        None
                                    };
                                    render_jfa_mask(
                                        gl,
                                        options,
                                        mask_fbo,
                                        mask_id,
                                        bbx,
                                        bby,
                                        bbw,
                                        bbh,
                                        source_size.w,
                                        source_size.h,
                                        &mut self.jfa_pipeline,
                                        textures,
                                        rects_tex,
                                        &self.cached_rects_px,
                                        jfa_cache_hit,
                                        previous_bbox,
                                    );
                                    self.cached_jfa_mask_texture_id = Some(mask_id);
                                }
                                _ => {
                                    warn!("RegionVectors mask pass: no valid bbox/subregion rects");
                                }
                            }
                        }

                        MaskPassStep::Custom { .. } => {
                            if let Some(Some(prog)) = mask_programs.get(i) {
                                if let (Some(rects_tex), true) = (
                                    self.rects_texture.as_ref(),
                                    !options.subregion_rects.is_empty(),
                                ) {
                                    gl.ActiveTexture(ffi::TEXTURE1);
                                    gl.BindTexture(ffi::TEXTURE_2D, rects_tex.tex_id());
                                    gl.ActiveTexture(ffi::TEXTURE0);
                                }

                                gl.FramebufferTexture2D(
                                    ffi::DRAW_FRAMEBUFFER,
                                    ffi::COLOR_ATTACHMENT0,
                                    ffi::TEXTURE_2D,
                                    dst_mask.tex_id(),
                                    0,
                                );

                                gl.ActiveTexture(ffi::TEXTURE0);
                                gl.BindTexture(ffi::TEXTURE_2D, src_mask.tex_id());
                                gl.TexParameteri(
                                    ffi::TEXTURE_2D,
                                    ffi::TEXTURE_MIN_FILTER,
                                    ffi::LINEAR as i32,
                                );
                                gl.TexParameteri(
                                    ffi::TEXTURE_2D,
                                    ffi::TEXTURE_MAG_FILTER,
                                    ffi::LINEAR as i32,
                                );
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

                                gl.UseProgram(prog.program);
                                if prog.uniform_subregion_count >= 0 {
                                    gl.Uniform1i(
                                        prog.uniform_subregion_count,
                                        options.subregion_rects.len() as i32,
                                    );
                                }
                                if prog.uniform_subregion_rects >= 0 {
                                    gl.Uniform1i(prog.uniform_subregion_rects, 1);
                                }
                                if prog.uniform_output_size >= 0 {
                                    gl.Uniform2f(
                                        prog.uniform_output_size,
                                        mask_w as f32,
                                        mask_h as f32,
                                    );
                                }
                                if prog.uniform_bbox_origin >= 0 {
                                    gl.Uniform2f(prog.uniform_bbox_origin, 0.0, 0.0);
                                }
                                if prog.uniform_geo_size >= 0 {
                                    gl.Uniform2f(prog.uniform_geo_size, geo_size.0, geo_size.1);
                                }
                                if prog.uniform_corner_radius >= 0 {
                                    gl.Uniform4f(
                                        prog.uniform_corner_radius,
                                        corner_radius[0],
                                        corner_radius[1],
                                        corner_radius[2],
                                        corner_radius[3],
                                    );
                                }

                                gl.Viewport(0, 0, mask_w, mask_h);
                                draw_fullscreen_quad(gl, prog.attrib_vert);
                                check_gl_error(gl);

                                gl.ActiveTexture(ffi::TEXTURE0);
                                gl.BindTexture(ffi::TEXTURE_2D, dst_mask.tex_id());
                                gl.TexParameteri(
                                    ffi::TEXTURE_2D,
                                    ffi::TEXTURE_MIN_FILTER,
                                    ffi::LINEAR as i32,
                                );
                                gl.TexParameteri(
                                    ffi::TEXTURE_2D,
                                    ffi::TEXTURE_MAG_FILTER,
                                    ffi::LINEAR as i32,
                                );
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
                            }
                        }
                    }
                    if !matches!(step, MaskPassStep::RegionVectors)
                        && self.cached_jfa_mask_texture_id == Some(dst_mask.tex_id())
                    {
                        self.cached_jfa_mask_texture_id = None;
                    }

                    src_mask = dst_mask;
                    active_mask_is_b = !active_mask_is_b;
                }
            }
            if mask_cache_hit {
                active_mask_is_b = self.cached_mask_texture_is_b.unwrap();
            } else {
                self.cached_mask_rects = options.subregion_rects.clone();
                self.cached_mask_w = mask_w;
                self.cached_mask_h = mask_h;
                self.cached_mask_geo_size = geo_size;
                self.cached_mask_corner_radius = corner_radius;
                self.cached_mask_pipeline = Some(custom_program.0.clone());
                self.cached_mask_texture_is_b = Some(active_mask_is_b);
            }

            // Only advertise an exact GPU clip when the built-in program
            // that produced the final mask is available. A compile failure
            // keeps the existing CPU region-clipping fallback.
            if apply_output_mask {
                apply_output_mask = match mask_passes.last() {
                    None => self.binary_program.is_some(),
                    Some(MaskPassStep::RegionVectors) => self.jfa_pipeline.is_some(),
                    Some(_) => false,
                };
            }

            let active_mask = if active_mask_is_b { mask_b } else { mask_a };
            gl.ActiveTexture(ffi::TEXTURE1);
            gl.BindTexture(ffi::TEXTURE_2D, active_mask.tex_id());
            gl.ActiveTexture(ffi::TEXTURE0);

            // Step 2: render passes.

            for (i, step) in render_passes.iter().enumerate() {
                let (output_w, output_h) = pass_output_sizes[i];
                let (input_w, input_h) = if i == 0 {
                    (source_size.w, source_size.h)
                } else {
                    let prev = pass_output_sizes[i - 1];
                    (prev.0, prev.1)
                };

                match step {
                    RenderPassStep::DualKawaseBlur { passes, offset } => {
                        let dst = self.custom_textures[i].tex_id();
                        let src = if i == 0 {
                            source.tex_id()
                        } else {
                            self.custom_textures[i - 1].tex_id()
                        };

                        run_dual_kawase_blur(
                            gl,
                            &self.program,
                            src,
                            dst,
                            &self.textures,
                            options,
                            active_mask.tex_id(),
                            *passes,
                            *offset,
                            mask_fbo,
                        );
                    }

                    RenderPassStep::Custom { .. } => {
                        if let Some(Some(pass)) = render_programs.get(i) {
                            gl.UseProgram(pass.program);
                            gl.Uniform1i(pass.uniform_input, 0);
                            gl.Uniform2f(
                                pass.uniform_output_size,
                                output_w as f32,
                                output_h as f32,
                            );
                            gl.Uniform2f(pass.uniform_input_size, input_w as f32, input_h as f32);
                            gl.Uniform2f(
                                pass.uniform_half_pixel,
                                0.5 / output_w as f32,
                                0.5 / output_h as f32,
                            );
                            gl.Uniform1i(pass.uniform_pass, i as i32);
                            gl.Uniform1i(pass.uniform_pass_count, pass_count as i32);
                            gl.Uniform2f(pass.uniform_geo_size, geo_size.0, geo_size.1);
                            gl.Uniform4f(
                                pass.uniform_corner_radius,
                                corner_radius[0],
                                corner_radius[1],
                                corner_radius[2],
                                corner_radius[3],
                            );

                            if pass.uniform_mask >= 0 {
                                gl.Uniform1i(pass.uniform_mask, 1);
                            }

                            if pass.uniform_window_screen_rect >= 0 {
                                gl.Uniform4f(
                                    pass.uniform_window_screen_rect,
                                    options.window_screen_rect[0],
                                    options.window_screen_rect[1],
                                    options.window_screen_rect[2],
                                    options.window_screen_rect[3],
                                );
                            }

                            gl.Viewport(0, 0, output_w, output_h);

                            let dst = self.custom_textures[i].tex_id();
                            let src = if i == 0 {
                                source.tex_id()
                            } else {
                                self.custom_textures[i - 1].tex_id()
                            };

                            trace!("custom blur pass {i}: drawing {src} to {dst}");

                            gl.FramebufferTexture2D(
                                ffi::DRAW_FRAMEBUFFER,
                                ffi::COLOR_ATTACHMENT0,
                                ffi::TEXTURE_2D,
                                dst,
                                0,
                            );

                            gl.BindTexture(ffi::TEXTURE_2D, src);
                            gl.TexParameteri(
                                ffi::TEXTURE_2D,
                                ffi::TEXTURE_MIN_FILTER,
                                ffi::LINEAR as i32,
                            );
                            gl.TexParameteri(
                                ffi::TEXTURE_2D,
                                ffi::TEXTURE_MAG_FILTER,
                                ffi::LINEAR as i32,
                            );
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

                            draw_fullscreen_quad(gl, pass.attrib_vert);
                        }
                    }
                }
            }

            if apply_output_mask && self.mask_output_program.is_none() {
                match compile_mask_output_program(gl) {
                    Ok(program) => self.mask_output_program = Some(program),
                    Err(err) => {
                        warn!("error compiling exact output mask shader: {err:?}");
                        apply_output_mask = false;
                    }
                }
            }

            if apply_output_mask {
                let program = self.mask_output_program.as_ref().unwrap();
                let input = self.custom_textures.last().unwrap();
                let output = self.masked_output_texture.as_ref().unwrap();

                gl.UseProgram(program.program);
                gl.Uniform1i(program.uniform_input, 0);
                gl.Uniform1i(program.uniform_mask, 1);
                gl.Viewport(0, 0, final_output_w, final_output_h);
                gl.FramebufferTexture2D(
                    ffi::DRAW_FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    output.tex_id(),
                    0,
                );

                gl.ActiveTexture(ffi::TEXTURE0);
                gl.BindTexture(ffi::TEXTURE_2D, input.tex_id());
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

                gl.ActiveTexture(ffi::TEXTURE1);
                gl.BindTexture(ffi::TEXTURE_2D, active_mask.tex_id());
                gl.TexParameteri(
                    ffi::TEXTURE_2D,
                    ffi::TEXTURE_MIN_FILTER,
                    ffi::NEAREST as i32,
                );
                gl.TexParameteri(
                    ffi::TEXTURE_2D,
                    ffi::TEXTURE_MAG_FILTER,
                    ffi::NEAREST as i32,
                );
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

                draw_fullscreen_quad(gl, program.attrib_vert);
                gl.ActiveTexture(ffi::TEXTURE0);
            }
            gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);
            gl.DeleteFramebuffers(1, &mask_fbo);
            check_gl_error(gl);
        })?;

        let texture = if apply_output_mask {
            self.masked_output_texture.as_ref().unwrap().clone()
        } else {
            self.custom_textures.last().unwrap().clone()
        };
        Ok(BlurOutput {
            texture,
            subregion_clipped: apply_output_mask,
        })
    }
}

fn render_jfa_mask(
    gl: &ffi::Gles2,
    options: &BlurOptions,
    mask_fbo: ffi::types::GLuint,
    mask_tex_id: ffi::types::GLuint,
    bbx: i32,
    bby: i32,
    bbw: i32,
    bbh: i32,
    source_w: i32,
    source_h: i32,
    jfa_pipeline: &mut Option<JfaPipeline>,
    textures: &JfaTextures,
    rects_tex: &GlesTexture,
    rects_px: &[[f32; 4]],
    cache_hit: bool,
    cache_previous_bbox: Option<(i32, i32, i32, i32)>,
) {
    unsafe {
        if cache_hit {
            if let Some((old_x, old_y, old_w, old_h)) = cache_previous_bbox {
                clear_mask_texture_region(gl, mask_fbo, mask_tex_id, old_x, old_y, old_w, old_h);
            } else {
                clear_mask_texture(gl, mask_fbo, mask_tex_id);
            }
            // textures.encoded is still valid from a previous frame
            // (bbox-local geometry hasn't changed), so we skip the JFA +
            blit_to_mask_texture(
                gl,
                &textures.encoded,
                mask_tex_id,
                bbw,
                bbh,
                source_w,
                source_h,
                rects_px,
                bbx,
                bby,
            );
            gl.BindTexture(ffi::TEXTURE_2D, mask_tex_id);
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
            return;
        }

        clear_mask_texture(gl, mask_fbo, mask_tex_id);

        let pipeline = match ensure_jfa_pipeline(gl, jfa_pipeline) {
            Ok(()) => jfa_pipeline.as_ref().unwrap(),
            Err(err) => {
                warn!("error compiling JFA shaders: {err:?}");
                return;
            }
        };

        render_binary_mask(
            gl,
            &pipeline.binary_prog,
            options,
            &textures.bin,
            rects_tex,
            bbx,
            bby,
            bbw,
            bbh,
        );
        jfa_init_pass(
            gl,
            &pipeline.init_prog,
            &textures.bin,
            &textures.jfa_a,
            bbw,
            bbh,
        );
        let jfa_result = run_jfa_steps(
            gl,
            &pipeline.step_prog,
            &textures.jfa_a,
            &textures.jfa_b,
            bbw,
            bbh,
        );
        // The normalization scale is shared by the encoded mask and its
        // smooth Poisson direction field.
        let max_dist = (max(1, min(bbw, bbh)) as f32) / 2.0;

        // Multigrid Poisson solve for the direction field.
        // 1. Write level-0 RHS from binary mask.
        // 2. Restrict the mask through the pyramid.
        // 3. Zero all u_a textures (initial guess u = 0).
        // 4. Run one V-cycle.
        // Scale the RHS so the Poisson solution stays in a
        // precision-friendly range (peak u ~ 1/16 instead of ~bbox²/16).
        // The encode pass multiplies the gradient back by max_dist.
        let f_scale = 1.0 / (max_dist * max_dist);
        init_rhs_level0(
            gl,
            &pipeline.poisson_init_rhs_prog,
            &textures.bin,
            &textures.pyramid[0].rhs,
            textures.pyramid[0].size.w,
            textures.pyramid[0].size.h,
            f_scale,
        );
        restrict_mask_pyramid(gl, &pipeline.poisson_restrict_mask_prog, &textures.pyramid);
        for lvl in &textures.pyramid {
            gl.FramebufferTexture2D(
                ffi::DRAW_FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                lvl.u_a.tex_id(),
                0,
            );
            gl.ClearColor(0.0, 0.0, 0.0, 0.0);
            gl.Clear(ffi::COLOR_BUFFER_BIT);
        }
        // A deep V-cycle resolves the low-frequency shape. Keep the per-level
        // Jacobi work modest; the encode pass smooths the differentiated
        // direction field directly instead of oversolving u to hide sampling
        // artifacts.
        let mut final_u_idx = 0;
        for _ in 0..V_CYCLES {
            final_u_idx = run_v_cycle(
                gl,
                pipeline,
                &textures.pyramid,
                JACOBI_SWEEPS,
                JACOBI_SWEEPS,
                JACOBI_OMEGA,
                final_u_idx,
            );
        }
        let poisson_u = if final_u_idx == 0 {
            &textures.pyramid[0].u_a
        } else {
            &textures.pyramid[0].u_b
        };

        // Encode into the JFA ping-pong texture that no longer holds the
        // nearest-seed result, then low-pass the encoded direction into the
        // persistent cache texture. Filtering after normalization removes
        // the block boundaries that filtering u alone cannot.
        let encode_scratch = if jfa_result.tex_id() == textures.jfa_a.tex_id() {
            &textures.jfa_b
        } else {
            &textures.jfa_a
        };
        encode_output(
            gl,
            &pipeline.encode_prog,
            jfa_result,
            poisson_u,
            encode_scratch,
            bbw,
            bbh,
            max_dist,
        );
        smooth_encoded_output(
            gl,
            &pipeline.smooth_prog,
            encode_scratch,
            &textures.encoded,
            bbw,
            bbh,
        );

        blit_to_mask_texture(
            gl,
            &textures.encoded,
            mask_tex_id,
            bbw,
            bbh,
            source_w,
            source_h,
            rects_px,
            bbx,
            bby,
        );

        check_gl_error(gl);

        // Set final mask sampler params.
        gl.BindTexture(ffi::TEXTURE_2D, mask_tex_id);
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
    }
}

unsafe fn clear_mask_texture(
    gl: &ffi::Gles2,
    fbo: ffi::types::GLuint,
    mask_tex_id: ffi::types::GLuint,
) {
    gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        mask_tex_id,
        0,
    );
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);
}

unsafe fn clear_mask_texture_region(
    gl: &ffi::Gles2,
    fbo: ffi::types::GLuint,
    mask_tex_id: ffi::types::GLuint,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) {
    if width <= 0 || height <= 0 {
        return;
    }
    gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        mask_tex_id,
        0,
    );
    gl.Enable(ffi::SCISSOR_TEST);
    gl.Scissor(x, y, width, height);
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);
    gl.Disable(ffi::SCISSOR_TEST);
}

unsafe fn ensure_jfa_pipeline(
    gl: &ffi::Gles2,
    jfa_pipeline: &mut Option<JfaPipeline>,
) -> anyhow::Result<()> {
    if jfa_pipeline.is_some() {
        return Ok(());
    }
    let pipeline = (|| -> Result<JfaPipeline, GlesError> {
        Ok(JfaPipeline {
            binary_prog: compile_binary_program(gl)?,
            init_prog: compile_jfa_init(gl)?,
            step_prog: compile_jfa_step(gl)?,
            poisson_init_rhs_prog: compile_jfa_poisson_init_rhs(gl)?,
            poisson_restrict_mask_prog: compile_jfa_poisson_restrict_mask(gl)?,
            poisson_jacobi_prog: compile_jfa_poisson_jacobi(gl)?,
            poisson_residual_restrict_prog: compile_jfa_poisson_residual_restrict(gl)?,
            poisson_prolongate_prog: compile_jfa_poisson_prolongate(gl)?,
            encode_prog: compile_jfa_encode(gl)?,
            smooth_prog: compile_jfa_smooth(gl)?,
        })
    })()
    .context("error compiling JFA shaders")?;
    *jfa_pipeline = Some(pipeline);
    Ok(())
}

unsafe fn render_binary_mask(
    gl: &ffi::Gles2,
    prog: &JfaBinaryProgram,
    options: &BlurOptions,
    dst: &GlesTexture,
    rects_tex: &GlesTexture,
    bbx: i32,
    bby: i32,
    bbw: i32,
    bbh: i32,
) {
    // The source-pixel rects are uploaded once by render_custom() and are
    // shared by the source binary, custom, and bbox-local JFA passes. Render
    // one quad instance per rectangle instead of making every destination
    // fragment scan the complete rectangle list.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, rects_tex.tex_id());

    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );
    gl.Disable(ffi::BLEND);
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_subregion_rects, 1);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Uniform2f(prog.uniform_bbox_origin, bbx as f32, bby as f32);

    gl.Viewport(0, 0, bbw, bbh);
    gl.Enable(ffi::BLEND);
    gl.BlendEquation(ffi::MAX);
    gl.DrawArraysInstanced(ffi::TRIANGLES, 0, 6, options.subregion_rects.len() as i32);
    gl.Disable(ffi::BLEND);
    gl.BlendEquation(ffi::FUNC_ADD);
    gl.ActiveTexture(ffi::TEXTURE0);
    // Keep the padded one-pixel border exterior even when a region touches
    // the source edge; JFA needs these exterior seeds.
    let border_x = BBOX_BORDER.min(bbw).max(0);
    let border_y = BBOX_BORDER.min(bbh).max(0);
    if border_x > 0 || border_y > 0 {
        gl.Enable(ffi::SCISSOR_TEST);
        gl.ClearColor(0.0, 0.0, 0.0, 0.0);
        if border_x > 0 {
            gl.Scissor(0, 0, border_x, bbh);
            gl.Clear(ffi::COLOR_BUFFER_BIT);
            gl.Scissor(bbw - border_x, 0, border_x, bbh);
            gl.Clear(ffi::COLOR_BUFFER_BIT);
        }
        if border_y > 0 {
            gl.Scissor(0, 0, bbw, border_y);
            gl.Clear(ffi::COLOR_BUFFER_BIT);
            gl.Scissor(0, bbh - border_y, bbw, border_y);
            gl.Clear(ffi::COLOR_BUFFER_BIT);
        }
        gl.Disable(ffi::SCISSOR_TEST);
    }
    gl.BindTexture(ffi::TEXTURE_2D, dst.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );
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
}

unsafe fn jfa_init_pass(
    gl: &ffi::Gles2,
    prog: &JfaInitProgram,
    src: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32,
    bbh: i32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);

    gl.Viewport(0, 0, bbw, bbh);
    gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
    draw_fullscreen_quad(gl, prog.attrib_vert);

    gl.BindTexture(ffi::TEXTURE_2D, dst.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );
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
}

unsafe fn run_jfa_steps<'a>(
    gl: &ffi::Gles2,
    prog: &JfaStepProgram,
    initial: &'a GlesTexture,
    scratch: &'a GlesTexture,
    bbw: i32,
    bbh: i32,
) -> &'a GlesTexture {
    let max_dim = max(bbw, bbh);
    let mut step = 1;
    while step * 2 <= max_dim {
        step *= 2;
    }

    let mut read_tex: &GlesTexture = initial;
    let mut write_tex: &GlesTexture = scratch;

    while step > 0 {
        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            write_tex.tex_id(),
            0,
        );

        gl.UseProgram(prog.program);
        gl.Uniform1i(prog.uniform_input, 0);
        gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
        gl.Uniform1i(prog.uniform_step, step);

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, read_tex.tex_id());
        gl.TexParameteri(
            ffi::TEXTURE_2D,
            ffi::TEXTURE_MIN_FILTER,
            ffi::NEAREST as i32,
        );
        gl.TexParameteri(
            ffi::TEXTURE_2D,
            ffi::TEXTURE_MAG_FILTER,
            ffi::NEAREST as i32,
        );

        draw_fullscreen_quad(gl, prog.attrib_vert);

        std::mem::swap(&mut read_tex, &mut write_tex);
        step /= 2;
    }

    // After the final swap, `read_tex` holds the most recent write.
    read_tex
}

// Writes pyramid[0].rhs from the binary mask:
//   R = 1 inside mask, 0 outside
//   G = mask (1 inside, 0 outside)
unsafe fn init_rhs_level0(
    gl: &ffi::Gles2,
    prog: &JfaPoissonInitRhsProgram,
    bin: &GlesTexture,
    dst_rhs: &GlesTexture,
    level_w: i32,
    level_h: i32,
    f_scale: f32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst_rhs.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.Uniform1f(prog.uniform_f_scale, f_scale);

    gl.Viewport(0, 0, level_w, level_h);
    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, bin.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );

    draw_fullscreen_quad(gl, prog.attrib_vert);
}

// For each pyramid level k = 1..pyramid.len(), box-downsamples the mask
// from level k-1's rhs into level k's rhs. Both R (`f`) and G (mask) are
// box-averaged; later residual_restrict overwrites R, but the mask in G
// is needed by the Jacobi smoother at all coarser levels.
unsafe fn restrict_mask_pyramid(
    gl: &ffi::Gles2,
    prog: &JfaPoissonRestrictMaskProgram,
    pyramid: &[MultigridLevel],
) {
    if pyramid.len() < 2 {
        return;
    }
    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.ActiveTexture(ffi::TEXTURE0);

    for k in 1..pyramid.len() {
        let src = &pyramid[k - 1];
        let dst = &pyramid[k];

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            dst.rhs.tex_id(),
            0,
        );

        gl.Uniform2f(
            prog.uniform_src_texel,
            1.0 / src.size.w as f32,
            1.0 / src.size.h as f32,
        );

        gl.BindTexture(ffi::TEXTURE_2D, src.rhs.tex_id());
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);

        gl.Viewport(0, 0, dst.size.w, dst.size.h);
        draw_fullscreen_quad(gl, prog.attrib_vert);
    }
}

// Runs `sweeps` weighted-Jacobi sweeps on pyramid[level], ping-ponging
// between u_a and u_b. Returns the index (0 = u_a, 1 = u_b) of the
// texture holding the latest u.
unsafe fn jacobi_smooth(
    gl: &ffi::Gles2,
    prog: &JfaPoissonJacobiProgram,
    pyramid: &[MultigridLevel],
    level: usize,
    sweeps: i32,
    current_u_idx: usize,
    omega: f32,
) -> usize {
    let lvl = &pyramid[level];
    let h_sq = (1u64 << (2 * level)) as f32; // h^2 = (2^level)^2 = 4^level

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_u_in, 0);
    gl.Uniform1i(prog.uniform_rhs, 1);
    gl.Uniform2f(
        prog.uniform_texel,
        1.0 / lvl.size.w as f32,
        1.0 / lvl.size.h as f32,
    );
    gl.Uniform1f(prog.uniform_h_sq, h_sq);
    gl.Uniform1f(prog.uniform_omega, omega);

    // RHS stays bound on TEXTURE1 across all sweeps.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, lvl.rhs.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );

    let mut current = current_u_idx;
    for _ in 0..sweeps {
        let (src, dst) = if current == 0 {
            (&lvl.u_a, &lvl.u_b)
        } else {
            (&lvl.u_b, &lvl.u_a)
        };

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            dst.tex_id(),
            0,
        );

        gl.ActiveTexture(ffi::TEXTURE0);
        gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
        gl.TexParameteri(
            ffi::TEXTURE_2D,
            ffi::TEXTURE_MIN_FILTER,
            ffi::NEAREST as i32,
        );
        gl.TexParameteri(
            ffi::TEXTURE_2D,
            ffi::TEXTURE_MAG_FILTER,
            ffi::NEAREST as i32,
        );

        gl.Viewport(0, 0, lvl.size.w, lvl.size.h);
        draw_fullscreen_quad(gl, prog.attrib_vert);

        current = 1 - current;
    }
    current
}

// Computes residual at fine level and box-restricts it into the coarse
// level's rhs.R. The coarse mask (rhs.G) is preserved.
//
// Resets coarse-level u_a to zero (correction equation `L e = r` starts
// from e = 0). Returns nothing; caller assumes coarse u_idx = 0.
unsafe fn residual_restrict(
    gl: &ffi::Gles2,
    prog: &JfaPoissonResidualRestrictProgram,
    pyramid: &[MultigridLevel],
    fine_level: usize,
    fine_u_idx: usize,
) {
    let fine = &pyramid[fine_level];
    let coarse = &pyramid[fine_level + 1];
    let fine_u = if fine_u_idx == 0 {
        &fine.u_a
    } else {
        &fine.u_b
    };
    let h_sq_fine = (1u64 << (2 * fine_level)) as f32;

    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        coarse.rhs.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_u_fine, 0);
    gl.Uniform1i(prog.uniform_rhs_fine, 1);
    gl.Uniform2f(
        prog.uniform_fine_texel,
        1.0 / fine.size.w as f32,
        1.0 / fine.size.h as f32,
    );
    gl.Uniform1f(prog.uniform_h_sq_fine, h_sq_fine);

    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, fine_u.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );

    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, fine.rhs.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );

    gl.Viewport(0, 0, coarse.size.w, coarse.size.h);

    // Restrict only the R channel; G (mask) was populated once per frame
    // by restrict_mask_pyramid and must not be touched here, otherwise
    // we'd be reading and writing the same texture (undefined per GL
    // spec; observed driver behaviour was unstable).
    gl.ColorMask(ffi::TRUE, ffi::FALSE, ffi::FALSE, ffi::FALSE);
    draw_fullscreen_quad(gl, prog.attrib_vert);
    gl.ColorMask(ffi::TRUE, ffi::TRUE, ffi::TRUE, ffi::TRUE);
    gl.ActiveTexture(ffi::TEXTURE0);

    // Reset coarse u_a to zero so subsequent smoothing solves the
    // correction equation from e = 0.
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        coarse.u_a.tex_id(),
        0,
    );
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);
}

// Bilinear-upsamples the coarse-level correction and adds it into the
// fine-level solution. Writes into the OPPOSITE of `fine_u_idx`; returns
// the new fine u idx.
unsafe fn prolongate(
    gl: &ffi::Gles2,
    prog: &JfaPoissonProlongateProgram,
    pyramid: &[MultigridLevel],
    fine_level: usize,
    fine_u_idx: usize,
    coarse_u_idx: usize,
) -> usize {
    let fine = &pyramid[fine_level];
    let coarse = &pyramid[fine_level + 1];
    let fine_u = if fine_u_idx == 0 {
        &fine.u_a
    } else {
        &fine.u_b
    };
    let fine_dst = if fine_u_idx == 0 {
        &fine.u_b
    } else {
        &fine.u_a
    };
    let coarse_u = if coarse_u_idx == 0 {
        &coarse.u_a
    } else {
        &coarse.u_b
    };

    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        fine_dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_u_fine, 0);
    gl.Uniform1i(prog.uniform_correction, 1);
    gl.Uniform1i(prog.uniform_rhs_fine, 2);

    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, fine_u.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );

    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, coarse_u.tex_id());
    // LINEAR upsampling for the coarse correction — this is the bilinear
    // prolongation operator.
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

    gl.ActiveTexture(ffi::TEXTURE2);
    gl.BindTexture(ffi::TEXTURE_2D, fine.rhs.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );

    gl.Viewport(0, 0, fine.size.w, fine.size.h);
    draw_fullscreen_quad(gl, prog.attrib_vert);
    gl.ActiveTexture(ffi::TEXTURE0);

    1 - fine_u_idx
}

// Runs one full V-cycle on the multigrid pyramid. Returns the index
// (0 or 1) of the level-0 `u_a`/`u_b` texture that contains the final
// solution.
//
// Caller must have:
//   1. Initialised pyramid[0].rhs via init_rhs_level0.
//   2. Restricted the mask through the pyramid via restrict_mask_pyramid.
//   3. Cleared all pyramid[k].u_a to zero (initial guess u = 0 for k=0, correction initial guess e
//      = 0 for k > 0).
unsafe fn run_v_cycle(
    gl: &ffi::Gles2,
    pipeline: &JfaPipeline,
    pyramid: &[MultigridLevel],
    n_pre: i32,
    n_post: i32,
    omega: f32,
    initial_u_idx: usize,
) -> usize {
    let k_max = pyramid.len() - 1;
    let mut u_idx: Vec<usize> = vec![0; pyramid.len()];
    u_idx[0] = initial_u_idx;

    // Down sweep.
    for k in 0..k_max {
        u_idx[k] = jacobi_smooth(
            gl,
            &pipeline.poisson_jacobi_prog,
            pyramid,
            k,
            n_pre,
            u_idx[k],
            omega,
        );
        residual_restrict(
            gl,
            &pipeline.poisson_residual_restrict_prog,
            pyramid,
            k,
            u_idx[k],
        );
        // residual_restrict zeroed coarse.u_a, so the coarse level
        // starts from u_idx = 0.
        u_idx[k + 1] = 0;
    }

    // Coarsest level — over-smooth as a stand-in for direct solve.
    u_idx[k_max] = jacobi_smooth(
        gl,
        &pipeline.poisson_jacobi_prog,
        pyramid,
        k_max,
        2 * n_pre,
        u_idx[k_max],
        omega,
    );

    // Up sweep.
    for k in (0..k_max).rev() {
        u_idx[k] = prolongate(
            gl,
            &pipeline.poisson_prolongate_prog,
            pyramid,
            k,
            u_idx[k],
            u_idx[k + 1],
        );
        u_idx[k] = jacobi_smooth(
            gl,
            &pipeline.poisson_jacobi_prog,
            pyramid,
            k,
            n_post,
            u_idx[k],
            omega,
        );
    }

    u_idx[0]
}

unsafe fn encode_output(
    gl: &ffi::Gles2,
    prog: &JfaEncodeProgram,
    jfa: &GlesTexture,
    poisson_u: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32,
    bbh: i32,
    max_dist: f32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.Uniform1i(prog.uniform_poisson_u, 1);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Uniform1f(prog.uniform_max_dist, max_dist);

    gl.Viewport(0, 0, bbw, bbh);

    // Poisson u on TEXTURE1 with NEAREST sampling for exact central
    // differences. Direction smoothing happens after normalization.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, poisson_u.tex_id());
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MIN_FILTER,
        ffi::NEAREST as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_MAG_FILTER,
        ffi::NEAREST as i32,
    );
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
    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, jfa.tex_id());

    draw_fullscreen_quad(gl, prog.attrib_vert);
}

unsafe fn smooth_encoded_output(
    gl: &ffi::Gles2,
    prog: &JfaSmoothProgram,
    src: &GlesTexture,
    dst: &GlesTexture,
    bbw: i32,
    bbh: i32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(prog.uniform_input, 0);
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Viewport(0, 0, bbw, bbh);

    gl.ActiveTexture(ffi::TEXTURE0);
    gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
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

    draw_fullscreen_quad(gl, prog.attrib_vert);
}

unsafe fn blit_to_mask_texture(
    gl: &ffi::Gles2,
    src_encoded: &GlesTexture,
    mask_tex_id: ffi::types::GLuint,
    bbw: i32,
    bbh: i32,
    source_w: i32,
    source_h: i32,
    rects_px: &[[f32; 4]],
    bbx: i32,
    bby: i32,
) {
    // Blit dest: union of region rects in source pixels (clamped to source bounds).
    let blit_x1 = rects_px
        .iter()
        .map(|r| r[0] as i32)
        .min()
        .unwrap_or(bbx + BBOX_BORDER)
        .max(0);
    let blit_y1 = rects_px
        .iter()
        .map(|r| r[1] as i32)
        .min()
        .unwrap_or(bby + BBOX_BORDER)
        .max(0);
    let blit_x2 = rects_px
        .iter()
        .map(|r| r[2] as i32)
        .max()
        .unwrap_or(bbx + bbw - BBOX_BORDER)
        .min(source_w);
    let blit_y2 = rects_px
        .iter()
        .map(|r| r[3] as i32)
        .max()
        .unwrap_or(bby + bbh - BBOX_BORDER)
        .min(source_h);

    let mut read_fbo = 0u32;
    gl.GenFramebuffers(1, &mut read_fbo);
    gl.BindFramebuffer(ffi::READ_FRAMEBUFFER, read_fbo);
    gl.FramebufferTexture2D(
        ffi::READ_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        src_encoded.tex_id(),
        0,
    );

    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        mask_tex_id,
        0,
    );

    gl.BlitFramebuffer(
        BBOX_BORDER,
        BBOX_BORDER,
        bbw - BBOX_BORDER,
        bbh - BBOX_BORDER,
        blit_x1,
        blit_y1,
        blit_x2,
        blit_y2,
        ffi::COLOR_BUFFER_BIT,
        ffi::LINEAR,
    );

    gl.DeleteFramebuffers(1, &read_fbo);
}

/// Run the full dual-Kawase down+up blur chain, writing the result
/// to `dst_tex`.  Uses `blur.textures` for intermediate storage; callers
/// must ensure `ensure_kawase_textures` has been called first.
///
/// The active mask texture ID is exposed on TEXTURE1 throughout, so
/// mask-aware variants of the blur shaders can sample it.  The built-in
/// `blur_down.frag`/`blur_up.frag` currently ignore it.
unsafe fn run_dual_kawase_blur(
    gl: &ffi::Gles2,
    program: &BlurProgram,
    src_tex: u32,
    dst_tex: u32,
    textures: &[GlesTexture],
    options: &BlurOptions,
    mask_tex_id: u32,
    passes_override: Option<u8>,
    offset_override: Option<f32>,
    fbo: ffi::types::GLuint,
) {
    let passes = passes_override
        .map(|p| p.clamp(1, MAX_KAWASE_PASSES) as usize)
        .unwrap_or_else(|| options.clamped_passes());
    let offset = offset_override.unwrap_or(options.offset as f32);

    gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbo);

    debug_assert!(
        textures.len() >= passes + 1,
        "run_dual_kawase_blur: textures ({}) < passes+1 ({})",
        textures.len(),
        passes + 1
    );

    // Bind mask on TEXTURE1 for the entire chain (shader may ignore it).
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, mask_tex_id);
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
    gl.ActiveTexture(ffi::TEXTURE0);

    let down_prog = &program.0.down;
    gl.UseProgram(down_prog.program);
    gl.Uniform1i(down_prog.uniform_tex, 0);
    gl.Uniform1f(down_prog.uniform_offset, offset);

    gl.EnableVertexAttribArray(down_prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        down_prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        FULLSCREEN_TRI_STRIP.as_ptr().cast(),
    );

    // Down passes: source → textures[1] → textures[2] → …
    // textures[0] is full-size (reserved for final up output) but we
    // write the final up result to dst_tex instead.
    // Down chain: src_tex → textures[1] → textures[2] → … → textures[passes]
    {
        let mut last_id = src_tex;
        for i in 0..passes {
            let dst = &textures[i + 1];
            let dst_size = dst.size();
            let w = dst_size.w;
            let h = dst_size.h;
            gl.Viewport(0, 0, w, h);
            gl.Uniform2f(down_prog.uniform_half_pixel, 0.5 / w as f32, 0.5 / h as f32);

            gl.FramebufferTexture2D(
                ffi::DRAW_FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                dst.tex_id(),
                0,
            );

            gl.ActiveTexture(ffi::TEXTURE0);
            gl.BindTexture(ffi::TEXTURE_2D, last_id);
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

            gl.DrawArrays(ffi::TRIANGLES, 0, 6);

            last_id = dst.tex_id();
        }
    }

    gl.DisableVertexAttribArray(down_prog.attrib_vert as u32);

    // Up passes: textures[passes] → … → textures[1] → dst_tex
    let up_prog = &program.0.up;
    gl.UseProgram(up_prog.program);
    gl.Uniform1i(up_prog.uniform_tex, 0);
    gl.Uniform1f(up_prog.uniform_offset, offset);

    gl.EnableVertexAttribArray(up_prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        up_prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        FULLSCREEN_TRI_STRIP.as_ptr().cast(),
    );

    {
        // Each up pass doubles the resolution.
        // src_chain: [tex3(eighth), tex2(quarter), tex1(half)]
        // dst_ids:   [tex2_id,       tex1_id,       dst_tex]
        // dst_sizes: [quarter,       half,          full]
        let src_chain: Vec<&GlesTexture> = textures.iter().skip(1).rev().collect();
        let first_dst_size = textures[0].size();
        let dst_ids: Vec<u32> = textures[1..]
            .iter()
            .rev()
            .skip(1)
            .map(|t| t.tex_id())
            .chain(std::iter::once(dst_tex))
            .collect();
        let dst_sizes: Vec<Size<i32, Buffer>> = textures[1..]
            .iter()
            .rev()
            .skip(1)
            .map(|t| t.size())
            .chain(std::iter::once(first_dst_size))
            .collect();

        for ((src, &dst_id), dst_size) in src_chain.iter().zip(dst_ids.iter()).zip(dst_sizes.iter())
        {
            gl.Viewport(0, 0, dst_size.w, dst_size.h);

            let src_size = src.size();
            gl.Uniform2f(
                up_prog.uniform_half_pixel,
                0.5 / src_size.w as f32,
                0.5 / src_size.h as f32,
            );

            gl.FramebufferTexture2D(
                ffi::DRAW_FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                dst_id,
                0,
            );

            gl.ActiveTexture(ffi::TEXTURE0);
            gl.BindTexture(ffi::TEXTURE_2D, src.tex_id());
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

            gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        }
    }

    gl.DisableVertexAttribArray(up_prog.attrib_vert as u32);
}

impl Blur {
    pub fn render(
        &mut self,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        options: &BlurOptions,
    ) -> anyhow::Result<BlurOutput> {
        let _span = tracy_client::span!("Blur::render");
        trace!("rendering blur");

        // Per-window custom blur pipeline: if `options.shader_pipeline` is
        // set, look up (or compile-on-first-use) the corresponding
        // pipeline from the cache. Falls through to the default Kawase
        // blur if not set or if the pipeline failed to compile.
        if let Some(ref pipeline) = options.shader_pipeline {
            let custom =
                crate::render_helpers::shaders::get_or_compile_custom_blur(renderer, pipeline);
            if let Some(custom) = custom {
                return self.render_custom(renderer, source, &custom, options);
            }
        }

        ensure!(
            renderer.context_id() == self.renderer_context_id,
            "wrong renderer"
        );

        let passes = options.clamped_passes();
        let size = source.size();

        // If shader_pipeline was set but compilation failed, prepare_textures()
        // skipped Kawase texture allocation. Allocate them here for the
        // fallback path.
        if self.textures.len() != passes + 1
            || self.textures.first().is_none_or(|t| t.size() != size)
        {
            self.ensure_kawase_textures(
                |fourcc, sz| renderer.create_buffer(fourcc, sz),
                source,
                passes,
            )?;
        }

        ensure!(
            self.textures.len() == passes + 1,
            "wrong textures len: expected {}, got {}",
            passes + 1,
            self.textures.len()
        );

        ensure!(
            self.textures[0].size() == size,
            "wrong output texture size: expected {size:?}, got {:?}",
            self.textures[0].size()
        );

        ensure!(
            self.textures[0].is_unique_reference(),
            "output texture has a non-unique reference"
        );

        renderer.with_profiled_context(gpu_span_location!("Blur::render"), |gl| unsafe {
            while gl.GetError() != ffi::NO_ERROR {}

            gl.Disable(ffi::BLEND);
            gl.Disable(ffi::SCISSOR_TEST);

            gl.ActiveTexture(ffi::TEXTURE0);

            let mut fbo = 0u32;
            gl.GenFramebuffers(1, &mut fbo);
            gl.BindFramebuffer(ffi::FRAMEBUFFER, fbo);

            let program = &self.program.0.down;
            gl.UseProgram(program.program);
            gl.Uniform1i(program.uniform_tex, 0);
            gl.Uniform1f(program.uniform_offset, options.offset as f32);

            gl.EnableVertexAttribArray(program.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                program.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                FULLSCREEN_TRI_STRIP.as_ptr().cast(),
            );

            let src = once(source).chain(&self.textures[1..]);
            let dst = &self.textures[1..];
            for (src, dst) in zip(src, dst) {
                let dst_size = dst.size();
                let w = dst_size.w;
                let h = dst_size.h;
                gl.Viewport(0, 0, w, h);

                // During downsampling, half_pixel is half of the destination pixel.
                gl.Uniform2f(program.uniform_half_pixel, 0.5 / w as f32, 0.5 / h as f32);

                let src = src.tex_id();
                let dst = dst.tex_id();

                trace!("drawing down {src} to {dst}");
                gl.FramebufferTexture2D(
                    ffi::FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    dst,
                    0,
                );

                gl.BindTexture(ffi::TEXTURE_2D, src);
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

                gl.DrawArrays(ffi::TRIANGLES, 0, 6);
            }

            gl.DisableVertexAttribArray(program.attrib_vert as u32);

            // Up
            let program = &self.program.0.up;
            gl.UseProgram(program.program);
            gl.Uniform1i(program.uniform_tex, 0);
            gl.Uniform1f(program.uniform_offset, options.offset as f32);

            gl.EnableVertexAttribArray(program.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                program.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                FULLSCREEN_TRI_STRIP.as_ptr().cast(),
            );

            let src = self.textures.iter().rev();
            let dst = self.textures.iter().rev().skip(1);
            for (src, dst) in zip(src, dst) {
                let dst_size = dst.size();
                let w = dst_size.w;
                let h = dst_size.h;
                gl.Viewport(0, 0, w, h);

                // During upsampling, half_pixel is half of the source pixel.
                let src_size = src.size();
                let src_w = src_size.w as f32;
                let src_h = src_size.h as f32;
                gl.Uniform2f(program.uniform_half_pixel, 0.5 / src_w, 0.5 / src_h);

                let src = src.tex_id();
                let dst = dst.tex_id();

                trace!("drawing up {src} to {dst}");
                gl.FramebufferTexture2D(
                    ffi::FRAMEBUFFER,
                    ffi::COLOR_ATTACHMENT0,
                    ffi::TEXTURE_2D,
                    dst,
                    0,
                );

                gl.BindTexture(ffi::TEXTURE_2D, src);
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

                gl.DrawArrays(ffi::TRIANGLES, 0, 6);
            }

            gl.DisableVertexAttribArray(program.attrib_vert as u32);

            gl.BindFramebuffer(ffi::FRAMEBUFFER, 0);
            gl.DeleteFramebuffers(1, &fbo);
            check_gl_error(gl);
        })?;

        Ok(BlurOutput {
            texture: self.textures[0].clone(),
            subregion_clipped: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_output_clip_requires_an_exact_builtin_region_mask() {
        assert!(Blur::can_gpu_clip_custom_output(&[], true));
        assert!(Blur::can_gpu_clip_custom_output(
            &[MaskPassStep::RegionVectors],
            true
        ));
        assert!(!Blur::can_gpu_clip_custom_output(
            &[MaskPassStep::WindowVectors],
            true
        ));
        assert!(!Blur::can_gpu_clip_custom_output(
            &[MaskPassStep::RegionVectors, MaskPassStep::WindowVectors],
            true
        ));
        assert!(!Blur::can_gpu_clip_custom_output(
            &[MaskPassStep::Custom {
                name: "mask".to_owned(),
                source: String::new(),
                scale: 1.0,
            }],
            true
        ));
        assert!(!Blur::can_gpu_clip_custom_output(
            &[MaskPassStep::RegionVectors],
            false
        ));
    }
}

use std::cmp::{max, min};
use std::iter::{once, zip};
use std::rc::Rc;

use anyhow::{ensure, Context as _};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::gles::{ffi, link_program, GlesError, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{ContextId, Offscreen as _, Renderer as _, Texture as _};
use smithay::gpu_span_location;
use smithay::utils::{Buffer, Size};

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
    /// Mask texture at source resolution, rendered from GPU mask shader.
    mask_texture: Option<GlesTexture>,
    /// Cached mask shader (compiled once per-blur-instance).
    mask_program: Option<MaskProgram>,
    /// Cache: last rects and size used for mask render.
    cached_mask_rects: Vec<[f32; 4]>,
    cached_mask_w: i32,
    cached_mask_h: i32,
    jfa_pipeline: Option<JfaPipeline>,
    jfa_textures: Option<JfaTextures>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct BlurOptions {
    pub passes: u8,
    pub offset: f64,
    pub geo_size: (f32, f32),
    pub corner_radius: [f32; 4],
    pub subregion_rects: Vec<[f32; 4]>,
    pub light_pos: (f32, f32),
    pub light_source: Option<niri_config::LightSource>,
}

impl BlurOptions {
    pub fn with_geometry(mut self, geo_size: (f32, f32), corner_radius: [f32; 4]) -> Self {
        self.geo_size = geo_size;
        self.corner_radius = corner_radius;
        self
    }

    pub fn with_subregion_rects(mut self, rects: Vec<[f32; 4]>) -> Self {
        self.subregion_rects = rects;
        self
    }

    pub fn with_light_pos(mut self, pos: (f32, f32)) -> Self {
        self.light_pos = pos;
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
            subregion_rects: Vec::new(),
            light_pos: (0.0, 0.0),
            light_source: config.light_source,
        }
    }
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
    uniform_light_pos: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct CustomBlurProgramInner {
    passes: Vec<CustomBlurPassProgram>,
    scales: Vec<f32>,
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
    let light_pos = c"niri_light_pos";
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
        uniform_light_pos: gl.GetUniformLocation(program, light_pos.as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, vert.as_ptr()),
    })
}

impl CustomBlurProgram {
    pub fn compile(
        renderer: &mut GlesRenderer,
        pass_configs: &[crate::render_helpers::custom_blur::CustomBlurPassConfig],
    ) -> anyhow::Result<Self> {
        let scales: Vec<f32> = pass_configs.iter().map(|c| c.scale).collect();
        renderer
            .with_context(move |gl| unsafe {
                let mut passes = Vec::with_capacity(pass_configs.len());
                for (i, config) in pass_configs.iter().enumerate() {
                    let pass = compile_custom_pass(gl, &config.source).with_context(|| {
                        format!("error compiling custom blur pass {} ({:?})", i, config.name)
                    })?;
                    passes.push(pass);
                }
                Ok(Self(Rc::new(CustomBlurProgramInner { passes, scales })))
            })
            .context("error making GL context current")?
    }

    pub fn destroy(self, renderer: &mut GlesRenderer) -> Result<(), GlesError> {
        renderer.with_context(move |gl| unsafe {
            for pass in &self.0.passes {
                gl.DeleteProgram(pass.program);
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
struct JfaBinaryProgram {
    program: ffi::types::GLuint,
    uniform_subregion_count: ffi::types::GLint,
    uniform_subregion_rects: ffi::types::GLint,
    uniform_mask_size: ffi::types::GLint,
    uniform_bbox_origin: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
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
struct JfaSdfBakeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
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
struct JfaPipeline {
    binary_prog: JfaBinaryProgram,
    init_prog: JfaInitProgram,
    step_prog: JfaStepProgram,
    sdf_bake_prog: JfaSdfBakeProgram,
    poisson_init_rhs_prog: JfaPoissonInitRhsProgram,
    poisson_restrict_mask_prog: JfaPoissonRestrictMaskProgram,
    poisson_jacobi_prog: JfaPoissonJacobiProgram,
    poisson_residual_restrict_prog: JfaPoissonResidualRestrictProgram,
    poisson_prolongate_prog: JfaPoissonProlongateProgram,
    encode_prog: JfaEncodeProgram,
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
    /// Binary mask of the subregion union.
    bin: GlesTexture,
    /// JFA ping-pong A.
    jfa_a: GlesTexture,
    /// JFA ping-pong B.
    jfa_b: GlesTexture,
    /// Scalar SDF (R channel) baked from JFA; sampled by encode for R.
    sdf: GlesTexture,
    /// Multigrid pyramid. Index 0 is the finest (bbox-sized) level.
    pyramid: Vec<MultigridLevel>,
    /// Final encoded output (R=normalized SDF, GB=encoded direction).
    encoded: GlesTexture,
    /// Bbox size these textures were allocated for.
    size: Size<i32, Buffer>,
}

const MASK_VERTICES: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

// Allocates an RGBA32F GlesTexture directly via raw GL, bypassing
// smithay's `Fourcc → GL` mapping (which only knows RGBA8 / RGBA16F).
// Used for the multigrid `u` textures, where half-float precision is
// insufficient near gradient minima.
//
// Requires GL_EXT_color_buffer_float (for color-renderability) and
// GL_OES_texture_float_linear (only if the texture is sampled with
// LINEAR filtering — our prolongation pass does this on the coarse `u`,
// so the extension is required for correct V-cycle behaviour).
fn create_rgba32f_buffer(
    renderer: &mut GlesRenderer,
    size: Size<i32, Buffer>,
) -> Result<GlesTexture, GlesError> {
    let tex = renderer.with_context(|gl| unsafe {
        let mut tex = 0;
        gl.GenTextures(1, &mut tex);
        gl.BindTexture(ffi::TEXTURE_2D, tex);
        gl.TexImage2D(
            ffi::TEXTURE_2D,
            0,
            ffi::RGBA32F as i32,
            size.w,
            size.h,
            0,
            ffi::RGBA,
            ffi::FLOAT,
            std::ptr::null(),
        );
        tex
    })?;
    Ok(unsafe { GlesTexture::from_raw(renderer, Some(ffi::RGBA32F), false, tex, size) })
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

unsafe fn compile_jfa_binary(gl: &ffi::Gles2) -> Result<JfaBinaryProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/mask_binary.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaBinaryProgram {
        program,
        uniform_subregion_count: gl.GetUniformLocation(program, c"niri_subregion_count".as_ptr()),
        uniform_subregion_rects: gl.GetUniformLocation(program, c"niri_subregion_rects".as_ptr()),
        uniform_mask_size: gl.GetUniformLocation(program, c"niri_mask_size".as_ptr()),
        uniform_bbox_origin: gl.GetUniformLocation(program, c"niri_bbox_origin".as_ptr()),
        attrib_vert: gl.GetAttribLocation(program, c"vert".as_ptr()),
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

unsafe fn compile_jfa_sdf_bake(gl: &ffi::Gles2) -> Result<JfaSdfBakeProgram, GlesError> {
    let vert_src = include_str!("shaders/blur_custom.vert");
    let frag_src = include_str!("shaders/jfa_sdf_bake.frag");
    let program = unsafe { link_program(gl, vert_src, frag_src)? };
    Ok(JfaSdfBakeProgram {
        program,
        uniform_input: gl.GetUniformLocation(program, c"niri_input".as_ptr()),
        uniform_output_size: gl.GetUniformLocation(program, c"niri_output_size".as_ptr()),
        uniform_max_dist: gl.GetUniformLocation(program, c"niri_max_dist".as_ptr()),
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

impl Blur {
    pub fn new(renderer: &mut GlesRenderer) -> Option<Self> {
        let program = Shaders::get(renderer).blur.clone()?;
        Some(Self {
            program,
            renderer_context_id: renderer.context_id(),
            textures: Vec::new(),
            custom_textures: Vec::new(),
            mask_texture: None,
            mask_program: None,
            cached_mask_rects: Vec::new(),
            cached_mask_w: 0,
            cached_mask_h: 0,
            jfa_pipeline: None,
            jfa_textures: None,
        })
    }

    pub fn context_id(&self) -> ContextId<GlesTexture> {
        self.renderer_context_id.clone()
    }

    pub fn prepare_textures(
        &mut self,
        mut create_texture: impl FnMut(Fourcc, Size<i32, Buffer>) -> Result<GlesTexture, GlesError>,
        source: &GlesTexture,
        options: BlurOptions,
    ) -> anyhow::Result<()> {
        let _span = tracy_client::span!("Blur::prepare_textures");

        let passes = options.passes.clamp(1, 31) as usize;
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

    fn render_custom(
        &mut self,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        custom_program: &CustomBlurProgram,
        options: &BlurOptions,
    ) -> anyhow::Result<GlesTexture> {
        let _span = tracy_client::span!("Blur::render_custom");
        trace!("rendering custom blur");

        let geo_size = options.geo_size;
        let corner_radius = options.corner_radius;

        ensure!(
            renderer.context_id() == self.renderer_context_id,
            "wrong renderer"
        );

        let passes = &custom_program.0.passes;
        let scales = &custom_program.0.scales;
        let pass_count = passes.len();

        let source_size = source.size();
        let source_w = source_size.w as f32;
        let source_h = source_size.h as f32;

        let mut pass_output_sizes: Vec<(i32, i32)> = Vec::with_capacity(pass_count);
        let mut current_w = source_w;
        let mut current_h = source_h;
        for &scale in scales {
            current_w = (current_w * scale).max(1.0);
            current_h = (current_h * scale).max(1.0);
            pass_output_sizes.push((current_w as i32, current_h as i32));
        }

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

        let mask_w = source_size.w;
        let mask_h = source_size.h;

        let rects_changed = self.cached_mask_rects != options.subregion_rects;
        let size_changed = self.cached_mask_w != mask_w || self.cached_mask_h != mask_h;

        let render_mask = rects_changed || size_changed;

        if render_mask {
            trace!(
                "rendering GPU mask: {} rects, {}x{}",
                options.subregion_rects.len(),
                mask_w,
                mask_h,
            );
            self.cached_mask_rects = options.subregion_rects.clone();
            self.cached_mask_w = mask_w;
            self.cached_mask_h = mask_h;
        }

        let mask_size = Size::new(mask_w, mask_h);
        let need_new_mask = self
            .mask_texture
            .as_ref()
            .map_or(true, |t| t.size() != mask_size);
        if need_new_mask {
            let texture: GlesTexture = renderer.create_buffer(Fourcc::Abgr16161616f, mask_size)?;
            self.mask_texture = Some(texture);
        }

        let need_render = render_mask || need_new_mask;

        // Use JFA for explicit subregions; fall back to analytical SDF when
        // there's a single region covering the whole window (or none at all).
        let jfa_bbox = if need_render && !options.subregion_rects.is_empty() {
            let rects = &options.subregion_rects;
            let single_full = rects.len() == 1
                && rects[0][0] <= 0.001
                && rects[0][1] <= 0.001
                && rects[0][2] >= 0.999
                && rects[0][3] >= 0.999;
            if single_full {
                info!("single full-window subregion, using analytical SDF");
                None
            } else {
                let mut bbox = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
                for rect in &options.subregion_rects {
                    bbox[0] = bbox[0].min(rect[0]);
                    bbox[1] = bbox[1].min(rect[1]);
                    bbox[2] = bbox[2].max(rect[2]);
                    bbox[3] = bbox[3].max(rect[3]);
                }
                // Expand bbox by 1px on each side for a guaranteed exterior border.
                //
                // Subregion rects arrive as UV coords (f32) that round-tripped
                // through a divide-by-blur-width: an integer pixel coord like
                // 200 becomes 200/1920 in f32, then multiplied back by 1920
                // here gives 199.99999 or 200.00001 (±1 ULP). A naïve floor()
                // then jitters between 199 and 200 across frames, which
                // propagates to bbox-size oscillation and visible "wobble"
                // even though the geometric input is integer-stable.
                //
                // The ULP at value ~200 in f32 is ~2.4e-5. A 1e-4 epsilon is
                // an order of magnitude above ULP and well below 0.5, so it
                // snaps integer-close values to the integer without altering
                // genuinely fractional positions.
                const EPS: f32 = 1e-4;
                let bbx = ((bbox[0] * source_size.w as f32 + EPS).floor() as i32 - 1).max(0);
                let bby = ((bbox[1] * source_size.h as f32 + EPS).floor() as i32 - 1).max(0);
                let bbw = (((bbox[2] - bbox[0]) * source_size.w as f32 - EPS).ceil() as i32 + 2)
                    .max(1).min(source_size.w - bbx);
                let bbh = (((bbox[3] - bbox[1]) * source_size.h as f32 - EPS).ceil() as i32 + 2)
                    .max(1).min(source_size.h - bby);
                if bbw > 0 && bbh > 0 {
                    let bbox_size = Size::new(bbw, bbh);
                    let need_alloc = match &self.jfa_textures {
                        Some(t) => t.size != bbox_size,
                        None => true,
                    };
                    if need_alloc {
                        // Build the multigrid pyramid sized to the bbox.
                        // Level 0 is bbox_size; each subsequent level halves
                        // both dimensions (floor, min 1).
                        let pyramid_max_levels = 6usize;
                        let mut pyramid: Vec<MultigridLevel> =
                            Vec::with_capacity(pyramid_max_levels);
                        for k in 0..pyramid_max_levels {
                            let mw = std::cmp::max(1, bbw >> k);
                            let mh = std::cmp::max(1, bbh >> k);
                            let lvl_size = Size::new(mw, mh);
                            pyramid.push(MultigridLevel {
                                // u textures need 32-bit float precision; half-
                                // float bands the gradient near interior maxima.
                                u_a: create_rgba32f_buffer(renderer, lvl_size)?,
                                u_b: create_rgba32f_buffer(renderer, lvl_size)?,
                                rhs: renderer
                                    .create_buffer(Fourcc::Abgr16161616f, lvl_size)?,
                                size: lvl_size,
                            });
                            if mw == 1 || mh == 1 {
                                break;
                            }
                        }

                        self.jfa_textures = Some(JfaTextures {
                            bin: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_a: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            jfa_b: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            sdf: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            pyramid,
                            encoded: renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?,
                            size: bbox_size,
                        });
                    }
                    Some((bbx, bby, bbw, bbh))
                } else {
                    None
                }
            } // end single_full else
        } else {
            None
        };

        renderer.with_profiled_context(gpu_span_location!("Blur::render_custom"), |gl| unsafe {
            while gl.GetError() != ffi::NO_ERROR {}

            gl.Disable(ffi::BLEND);
            gl.Disable(ffi::SCISSOR_TEST);
            gl.ActiveTexture(ffi::TEXTURE0);

            if let Some(mask_tex) = &self.mask_texture {
                if need_render {
                    if let Some((bbx, bby, bbw, bbh)) = jfa_bbox {
                        let mask_tex_id = mask_tex.tex_id();
                        let textures = self.jfa_textures.as_ref().unwrap();
                        render_jfa_mask(
                            gl,
                            options,
                            mask_tex_id,
                            bbx,
                            bby,
                            bbw,
                            bbh,
                            source_size.w,
                            source_size.h,
                            &mut self.jfa_pipeline,
                            textures,
                        );
                    } else {
                        // Compile mask shader lazily, cache for reuse.
                        if self.mask_program.is_none() {
                            match compile_mask_program(gl) {
                                Ok(p) => self.mask_program = Some(p),
                                Err(err) => {
                                    warn!("error compiling mask shader: {err:?}");
                                    return;
                                }
                            }
                        }
                        let mask_prog = self.mask_program.as_ref().unwrap();

                        let mut mask_fbo = 0u32;
                        gl.GenFramebuffers(1, &mut mask_fbo);
                        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, mask_fbo);
                        gl.FramebufferTexture2D(
                            ffi::DRAW_FRAMEBUFFER,
                            ffi::COLOR_ATTACHMENT0,
                            ffi::TEXTURE_2D,
                            mask_tex.tex_id(),
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
                        gl.EnableVertexAttribArray(mask_prog.attrib_vert as u32);
                        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
                        gl.VertexAttribPointer(
                            mask_prog.attrib_vert as u32,
                            2,
                            ffi::FLOAT,
                            ffi::FALSE,
                            0,
                            MASK_VERTICES.as_ptr().cast(),
                        );

                        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
                        gl.DisableVertexAttribArray(mask_prog.attrib_vert as u32);

                        gl.DeleteFramebuffers(1, &mut mask_fbo);
                        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);

                        // Set texture params on the rendered mask.
                        gl.BindTexture(ffi::TEXTURE_2D, mask_tex.tex_id());
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

                gl.ActiveTexture(ffi::TEXTURE1);
                gl.BindTexture(ffi::TEXTURE_2D, mask_tex.tex_id());
                gl.ActiveTexture(ffi::TEXTURE0);
            }

            let mut fbos = [0; 2];
            gl.GenFramebuffers(fbos.len() as _, fbos.as_mut_ptr());
            gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbos[0]);

            let vertices = MASK_VERTICES;

            for (i, pass) in passes.iter().enumerate() {
                let (output_w, output_h) = pass_output_sizes[i];
                let (input_w, input_h) = if i == 0 {
                    (source_size.w, source_size.h)
                } else {
                    let prev = pass_output_sizes[i - 1];
                    (prev.0, prev.1)
                };

                gl.UseProgram(pass.program);
                gl.Uniform1i(pass.uniform_input, 0);
                gl.Uniform2f(pass.uniform_output_size, output_w as f32, output_h as f32);
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

                if pass.uniform_light_pos >= 0 {
                    gl.Uniform2f(
                        pass.uniform_light_pos,
                        options.light_pos.0,
                        options.light_pos.1,
                    );
                }

                gl.Viewport(0, 0, output_w, output_h);

                gl.EnableVertexAttribArray(pass.attrib_vert as u32);
                gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
                gl.VertexAttribPointer(
                    pass.attrib_vert as u32,
                    2,
                    ffi::FLOAT,
                    ffi::FALSE,
                    0,
                    vertices.as_ptr().cast(),
                );

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
                gl.DisableVertexAttribArray(pass.attrib_vert as u32);
            }

            gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);
            gl.DeleteFramebuffers(fbos.len() as _, fbos.as_ptr());
        })?;

        Ok(self.custom_textures.last().unwrap().clone())
    }
}

fn render_jfa_mask(
    gl: &ffi::Gles2,
    options: &BlurOptions,
    mask_tex_id: ffi::types::GLuint,
    bbx: i32,
    bby: i32,
    bbw: i32,
    bbh: i32,
    source_w: i32,
    source_h: i32,
    jfa_pipeline: &mut Option<JfaPipeline>,
    textures: &JfaTextures,
) {
    unsafe {
        clear_mask_texture(gl, mask_tex_id);
        if !ensure_jfa_pipeline(gl, jfa_pipeline) {
            return;
        }
        let pipeline = jfa_pipeline.as_ref().unwrap();

        let mut fbo = 0u32;
        gl.GenFramebuffers(1, &mut fbo);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbo);

        render_binary_mask(
            gl,
            &pipeline.binary_prog,
            options,
            &textures.bin,
            bbx,
            bby,
            bbw,
            bbh,
            source_w,
            source_h,
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
        // Compute the normalization scale once — used both by the SDF bake
        // (so the baked texture is in [0, 1]) and by encode_output (so the
        // mask R channel uses the same scale).
        let max_dist = (max(1, min(bbw, bbh)) as f32) / 2.0;

        bake_sdf(
            gl,
            &pipeline.sdf_bake_prog,
            jfa_result,
            &textures.sdf,
            bbw,
            bbh,
            max_dist,
        );

        // Multigrid Poisson solve for the direction field.
        // 1. Write level-0 RHS from binary mask.
        // 2. Restrict the mask through the pyramid.
        // 3. Zero all u_a textures (initial guess u = 0).
        // 4. Run one V-cycle.
        // Scale the RHS so the Poisson solution u stays in a
        // precision-friendly range for RGBA16F (peak u ~ 1/16 instead of
        // ~bbox²/16). The encode pass multiplies the gradient back by
        // max_dist to restore the [-1, 1] range expected by renderers.
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
        restrict_mask_pyramid(
            gl,
            &pipeline.poisson_restrict_mask_prog,
            &textures.pyramid,
        );
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
        // 2 V-cycles with 12 pre/post Jacobi sweeps each.
        //
        // Tuning notes: the visible artefacts in jfa-debug are mostly
        // sensitive to TOTAL fine-level smoothing (cycles × (n_pre + n_post)).
        // Raise sweep count if polygonal patches reappear at the centre of
        // the solid square or in deep interior regions. The two parameters
        // trade off: more cycles improves convergence rate (each cycle
        // reduces error by a constant factor), more sweeps improves
        // per-cycle damping.
        let mut final_u_idx = 0;
        for _ in 0..2 {
            final_u_idx =
                run_v_cycle(gl, pipeline, &textures.pyramid, 12, 12, 0.8, final_u_idx);
        }
        let poisson_u = if final_u_idx == 0 {
            &textures.pyramid[0].u_a
        } else {
            &textures.pyramid[0].u_b
        };

        encode_output(
            gl,
            &pipeline.encode_prog,
            jfa_result,
            poisson_u,
            &textures.encoded,
            bbw,
            bbh,
            max_dist,
        );

        blit_to_mask_texture(
            gl,
            &textures.encoded,
            mask_tex_id,
            bbw,
            bbh,
            source_w,
            source_h,
            options,
            bbx,
            bby,
        );

        gl.DeleteFramebuffers(1, &mut fbo);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);

        // Set final mask sampler params.
        gl.BindTexture(ffi::TEXTURE_2D, mask_tex_id);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::LINEAR as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
    }
}

unsafe fn clear_mask_texture(gl: &ffi::Gles2, mask_tex_id: ffi::types::GLuint) {
    let mut clear_fbo = 0u32;
    gl.GenFramebuffers(1, &mut clear_fbo);
    gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, clear_fbo);
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        mask_tex_id,
        0,
    );
    gl.ClearColor(0.0, 0.0, 0.0, 0.0);
    gl.Clear(ffi::COLOR_BUFFER_BIT);
    gl.DeleteFramebuffers(1, &mut clear_fbo);
}

unsafe fn ensure_jfa_pipeline(gl: &ffi::Gles2, jfa_pipeline: &mut Option<JfaPipeline>) -> bool {
    if jfa_pipeline.is_some() {
        return true;
    }
    match (|| -> Result<JfaPipeline, GlesError> {
        Ok(JfaPipeline {
            binary_prog: compile_jfa_binary(gl)?,
            init_prog: compile_jfa_init(gl)?,
            step_prog: compile_jfa_step(gl)?,
            sdf_bake_prog: compile_jfa_sdf_bake(gl)?,
            poisson_init_rhs_prog: compile_jfa_poisson_init_rhs(gl)?,
            poisson_restrict_mask_prog: compile_jfa_poisson_restrict_mask(gl)?,
            poisson_jacobi_prog: compile_jfa_poisson_jacobi(gl)?,
            poisson_residual_restrict_prog: compile_jfa_poisson_residual_restrict(gl)?,
            poisson_prolongate_prog: compile_jfa_poisson_prolongate(gl)?,
            encode_prog: compile_jfa_encode(gl)?,
        })
    })() {
        Ok(p) => {
            *jfa_pipeline = Some(p);
            true
        }
        Err(err) => {
            warn!("error compiling JFA shaders: {err:?}");
            false
        }
    }
}

unsafe fn render_binary_mask(
    gl: &ffi::Gles2,
    prog: &JfaBinaryProgram,
    options: &BlurOptions,
    dst: &GlesTexture,
    bbx: i32,
    bby: i32,
    bbw: i32,
    bbh: i32,
    source_w: i32,
    source_h: i32,
) {
    gl.FramebufferTexture2D(
        ffi::DRAW_FRAMEBUFFER,
        ffi::COLOR_ATTACHMENT0,
        ffi::TEXTURE_2D,
        dst.tex_id(),
        0,
    );

    gl.UseProgram(prog.program);
    gl.Uniform1i(
        prog.uniform_subregion_count,
        options.subregion_rects.len() as i32,
    );

    // UV rects → source-pixel coords, then clamp inside the padded bbox border.
    let rects_px: Vec<[f32; 4]> = options
        .subregion_rects
        .iter()
        .map(|r| {
            [
                (r[0] * source_w as f32).max((bbx + 1) as f32),
                (r[1] * source_h as f32).max((bby + 1) as f32),
                (r[2] * source_w as f32).min((bbx + bbw - 2) as f32),
                (r[3] * source_h as f32).min((bby + bbh - 2) as f32),
            ]
        })
        .collect();

    gl.Uniform4fv(
        prog.uniform_subregion_rects,
        rects_px.len() as i32,
        rects_px.as_ptr() as *const f32,
    );
    gl.Uniform2f(prog.uniform_mask_size, bbw as f32, bbh as f32);
    gl.Uniform2f(prog.uniform_bbox_origin, bbx as f32, bby as f32);

    gl.Viewport(0, 0, bbw, bbh);
    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);

    gl.BindTexture(ffi::TEXTURE_2D, dst.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
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
    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);

    gl.BindTexture(ffi::TEXTURE_2D, dst.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::CLAMP_TO_EDGE as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::CLAMP_TO_EDGE as i32);
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
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

        gl.EnableVertexAttribArray(prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(prog.attrib_vert as u32);

        std::mem::swap(&mut read_tex, &mut write_tex);
        step /= 2;
    }

    // After the final swap, `read_tex` holds the most recent write.
    read_tex
}

// Bakes the JFA result into a scalar normalized SDF texture (R channel).
// dc / max_dist clamped to [0, 1]. Exterior pixels get 0.
unsafe fn bake_sdf(
    gl: &ffi::Gles2,
    prog: &JfaSdfBakeProgram,
    jfa: &GlesTexture,
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
    gl.Uniform2f(prog.uniform_output_size, bbw as f32, bbh as f32);
    gl.Uniform1f(prog.uniform_max_dist, max_dist);

    gl.Viewport(0, 0, bbw, bbh);
    gl.BindTexture(ffi::TEXTURE_2D, jfa.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
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
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
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

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );

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
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    }

    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
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
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
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
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
        gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

        gl.Viewport(0, 0, lvl.size.w, lvl.size.h);
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);

        current = 1 - current;
    }

    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
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
    let fine_u = if fine_u_idx == 0 { &fine.u_a } else { &fine.u_b };
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
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, fine.rhs.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );

    gl.Viewport(0, 0, coarse.size.w, coarse.size.h);

    // Restrict only the R channel; G (mask) was populated once per frame
    // by restrict_mask_pyramid and must not be touched here, otherwise
    // we'd be reading and writing the same texture (undefined per GL
    // spec; observed driver behaviour was unstable).
    gl.ColorMask(ffi::TRUE, ffi::FALSE, ffi::FALSE, ffi::FALSE);
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.ColorMask(ffi::TRUE, ffi::TRUE, ffi::TRUE, ffi::TRUE);

    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
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
    let fine_u = if fine_u_idx == 0 { &fine.u_a } else { &fine.u_b };
    let fine_dst = if fine_u_idx == 0 { &fine.u_b } else { &fine.u_a };
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
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

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
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );

    gl.Viewport(0, 0, fine.size.w, fine.size.h);
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
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
//   3. Cleared all pyramid[k].u_a to zero (initial guess u = 0 for k=0,
//      correction initial guess e = 0 for k > 0).
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

    // Poisson u on TEXTURE1 with NEAREST sampling — we want exact texel
    // values for the central-difference gradient.
    gl.ActiveTexture(ffi::TEXTURE1);
    gl.BindTexture(ffi::TEXTURE_2D, poisson_u.tex_id());
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, ffi::NEAREST as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, ffi::NEAREST as i32);
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

    gl.EnableVertexAttribArray(prog.attrib_vert as u32);
    gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
    gl.VertexAttribPointer(
        prog.attrib_vert as u32,
        2,
        ffi::FLOAT,
        ffi::FALSE,
        0,
        MASK_VERTICES.as_ptr().cast(),
    );
    gl.DrawArrays(ffi::TRIANGLES, 0, 6);
    gl.DisableVertexAttribArray(prog.attrib_vert as u32);
}

unsafe fn blit_to_mask_texture(
    gl: &ffi::Gles2,
    src_encoded: &GlesTexture,
    mask_tex_id: ffi::types::GLuint,
    bbw: i32,
    bbh: i32,
    source_w: i32,
    source_h: i32,
    options: &BlurOptions,
    bbx: i32,
    bby: i32,
) {
    // Blit dest: union of region rects in source pixels (clamped to source bounds).
    let rects_raw_px: Vec<[f32; 4]> = options
        .subregion_rects
        .iter()
        .map(|r| {
            [
                r[0] * source_w as f32,
                r[1] * source_h as f32,
                r[2] * source_w as f32,
                r[3] * source_h as f32,
            ]
        })
        .collect();

    let blit_x1 = rects_raw_px
        .iter()
        .map(|r| r[0] as i32)
        .min()
        .unwrap_or(bbx + 1)
        .max(0);
    let blit_y1 = rects_raw_px
        .iter()
        .map(|r| r[1] as i32)
        .min()
        .unwrap_or(bby + 1)
        .max(0);
    let blit_x2 = rects_raw_px
        .iter()
        .map(|r| r[2] as i32)
        .max()
        .unwrap_or(bbx + bbw - 1)
        .min(source_w);
    let blit_y2 = rects_raw_px
        .iter()
        .map(|r| r[3] as i32)
        .max()
        .unwrap_or(bby + bbh - 1)
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
        1,
        1,
        bbw - 1,
        bbh - 1,
        blit_x1,
        blit_y1,
        blit_x2,
        blit_y2,
        ffi::COLOR_BUFFER_BIT,
        ffi::LINEAR,
    );

    gl.DeleteFramebuffers(1, &mut read_fbo);
}

impl Blur {
    pub fn render(
        &mut self,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        options: BlurOptions,
    ) -> anyhow::Result<GlesTexture> {
        let _span = tracy_client::span!("Blur::render");
        trace!("rendering blur");

        let custom = Shaders::get(renderer).custom_blur.borrow().clone();
        if let Some(custom) = custom {
            return self.render_custom(renderer, source, &custom, &options);
        }

        ensure!(
            renderer.context_id() == self.renderer_context_id,
            "wrong renderer"
        );

        let passes = options.passes.clamp(1, 31) as usize;
        let size = source.size();

        ensure!(
            self.textures.len() == passes + 1,
            "wrong textures len: expected {}, got {}",
            passes + 1,
            self.textures.len()
        );

        let output = &mut self.textures[0];
        ensure!(
            output.size() == size,
            "wrong output texture size: expected {size:?}, got {:?}",
            output.size()
        );

        ensure!(
            output.is_unique_reference(),
            "output texture has a non-unique reference"
        );

        renderer.with_profiled_context(gpu_span_location!("Blur::render"), |gl| unsafe {
            while gl.GetError() != ffi::NO_ERROR {}

            gl.Disable(ffi::BLEND);
            gl.Disable(ffi::SCISSOR_TEST);

            gl.ActiveTexture(ffi::TEXTURE0);

            let mut fbos = [0; 2];
            gl.GenFramebuffers(fbos.len() as _, fbos.as_mut_ptr());
            gl.BindFramebuffer(ffi::FRAMEBUFFER, fbos[0]);

            let program = &self.program.0.down;
            gl.UseProgram(program.program);
            gl.Uniform1i(program.uniform_tex, 0);
            gl.Uniform1f(program.uniform_offset, options.offset as f32);

            let vertices: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
            gl.EnableVertexAttribArray(program.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                program.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                vertices.as_ptr().cast(),
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

            let vertices: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
            gl.EnableVertexAttribArray(program.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                program.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                vertices.as_ptr().cast(),
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
            gl.DeleteFramebuffers(fbos.len() as _, fbos.as_ptr());
        })?;

        Ok(self.textures[0].clone())
    }
}

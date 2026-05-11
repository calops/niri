use std::cmp::max;
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
    jfa_textures: Vec<GlesTexture>,
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
    uniform_half_pixel: ffi::types::GLint,
    uniform_step: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaEncodeProgram {
    program: ffi::types::GLuint,
    uniform_input: ffi::types::GLint,
    uniform_output_size: ffi::types::GLint,
    uniform_max_dist: ffi::types::GLint,
    attrib_vert: ffi::types::GLint,
}

#[derive(Debug)]
struct JfaPipeline {
    binary_prog: JfaBinaryProgram,
    init_prog: JfaInitProgram,
    step_prog: JfaStepProgram,
    encode_prog: JfaEncodeProgram,
}

const MASK_VERTICES: [f32; 12] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

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
        uniform_half_pixel: gl.GetUniformLocation(program, c"niri_half_pixel".as_ptr()),
        uniform_step: gl.GetUniformLocation(program, c"niri_step".as_ptr()),
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
            jfa_textures: Vec::new(),
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

        let jfa_bbox = if need_render && !options.subregion_rects.is_empty() {
            let mut bbox = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
            for rect in &options.subregion_rects {
                bbox[0] = bbox[0].min(rect[0]);
                bbox[1] = bbox[1].min(rect[1]);
                bbox[2] = bbox[2].max(rect[2]);
                bbox[3] = bbox[3].max(rect[3]);
            }
            let bbx = (bbox[0] * source_size.w as f32).floor() as i32;
            let bby = (bbox[1] * source_size.h as f32).floor() as i32;
            let bbw = ((bbox[2] - bbox[0]) * source_size.w as f32).ceil() as i32;
            let bbh = ((bbox[3] - bbox[1]) * source_size.h as f32).ceil() as i32;
            if bbw > 0 && bbh > 0 {
                let bbox_size = Size::new(bbw, bbh);
                self.jfa_textures.clear();
                for _ in 0..4 {
                    let tex = renderer.create_buffer(Fourcc::Abgr16161616f, bbox_size)?;
                    self.jfa_textures.push(tex);
                }
                Some((bbx, bby, bbw, bbh))
            } else {
                None
            }
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
                        render_jfa_mask(
                            gl,
                            options,
                            mask_tex_id,
                            bbx,
                            bby,
                            bbw,
                            bbh,
                            &mut self.jfa_pipeline,
                            &self.jfa_textures,
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
    jfa_pipeline: &mut Option<JfaPipeline>,
    jfa_textures: &[GlesTexture],
) {
    unsafe {
        if jfa_pipeline.is_none() {
            match (|| -> Result<JfaPipeline, GlesError> {
                Ok(JfaPipeline {
                    binary_prog: compile_jfa_binary(gl)?,
                    init_prog: compile_jfa_init(gl)?,
                    step_prog: compile_jfa_step(gl)?,
                    encode_prog: compile_jfa_encode(gl)?,
                })
            })() {
                Ok(p) => *jfa_pipeline = Some(p),
                Err(err) => {
                    warn!("error compiling JFA shaders: {err:?}");
                    return;
                }
            }
        }
        let pipeline = jfa_pipeline.as_ref().unwrap();
        let tex = jfa_textures;

        let mut fbo = 0u32;
        gl.GenFramebuffers(1, &mut fbo);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, fbo);

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            tex[0].tex_id(),
            0,
        );

        gl.UseProgram(pipeline.binary_prog.program);
        gl.Uniform1i(
            pipeline.binary_prog.uniform_subregion_count,
            options.subregion_rects.len() as i32,
        );
        gl.Uniform4fv(
            pipeline.binary_prog.uniform_subregion_rects,
            options.subregion_rects.len() as i32,
            options.subregion_rects.as_ptr() as *const f32,
        );
        gl.Uniform2f(
            pipeline.binary_prog.uniform_mask_size,
            bbw as f32,
            bbh as f32,
        );
        gl.Uniform2f(
            pipeline.binary_prog.uniform_bbox_origin,
            bbx as f32,
            bby as f32,
        );

        gl.Viewport(0, 0, bbw, bbh);
        gl.EnableVertexAttribArray(pipeline.binary_prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            pipeline.binary_prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(pipeline.binary_prog.attrib_vert as u32);

        gl.BindTexture(ffi::TEXTURE_2D, tex[0].tex_id());
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

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            tex[1].tex_id(),
            0,
        );

        gl.UseProgram(pipeline.init_prog.program);
        gl.Uniform1i(pipeline.init_prog.uniform_input, 0);
        gl.Uniform2f(
            pipeline.init_prog.uniform_output_size,
            bbw as f32,
            bbh as f32,
        );

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, tex[0].tex_id());
        gl.EnableVertexAttribArray(pipeline.init_prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            pipeline.init_prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(pipeline.init_prog.attrib_vert as u32);

        gl.BindTexture(ffi::TEXTURE_2D, tex[1].tex_id());
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

        let max_dim = max(bbw, bbh);
        let mut step = max_dim / 2;
        let mut read_idx = 1;
        let mut write_idx = 2;

        while step > 0 {
            gl.FramebufferTexture2D(
                ffi::DRAW_FRAMEBUFFER,
                ffi::COLOR_ATTACHMENT0,
                ffi::TEXTURE_2D,
                tex[write_idx].tex_id(),
                0,
            );

            gl.UseProgram(pipeline.step_prog.program);
            gl.Uniform1i(pipeline.step_prog.uniform_input, 0);
            gl.Uniform2f(
                pipeline.step_prog.uniform_output_size,
                bbw as f32,
                bbh as f32,
            );
            gl.Uniform2f(
                pipeline.step_prog.uniform_half_pixel,
                0.5 / bbw as f32,
                0.5 / bbh as f32,
            );
            gl.Uniform1i(pipeline.step_prog.uniform_step, step);

            gl.Viewport(0, 0, bbw, bbh);
            gl.BindTexture(ffi::TEXTURE_2D, tex[read_idx].tex_id());
            gl.EnableVertexAttribArray(pipeline.step_prog.attrib_vert as u32);
            gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
            gl.VertexAttribPointer(
                pipeline.step_prog.attrib_vert as u32,
                2,
                ffi::FLOAT,
                ffi::FALSE,
                0,
                MASK_VERTICES.as_ptr().cast(),
            );
            gl.DrawArrays(ffi::TRIANGLES, 0, 6);
            gl.DisableVertexAttribArray(pipeline.step_prog.attrib_vert as u32);

            gl.BindTexture(ffi::TEXTURE_2D, tex[write_idx].tex_id());
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

            std::mem::swap(&mut read_idx, &mut write_idx);
            step /= 2;
        }

        gl.FramebufferTexture2D(
            ffi::DRAW_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            tex[3].tex_id(),
            0,
        );

        gl.UseProgram(pipeline.encode_prog.program);
        gl.Uniform1i(pipeline.encode_prog.uniform_input, 0);
        gl.Uniform2f(
            pipeline.encode_prog.uniform_output_size,
            bbw as f32,
            bbh as f32,
        );
        gl.Uniform1f(pipeline.encode_prog.uniform_max_dist, max_dim as f32);

        gl.Viewport(0, 0, bbw, bbh);
        gl.BindTexture(ffi::TEXTURE_2D, tex[read_idx].tex_id());
        gl.EnableVertexAttribArray(pipeline.encode_prog.attrib_vert as u32);
        gl.BindBuffer(ffi::ARRAY_BUFFER, 0);
        gl.VertexAttribPointer(
            pipeline.encode_prog.attrib_vert as u32,
            2,
            ffi::FLOAT,
            ffi::FALSE,
            0,
            MASK_VERTICES.as_ptr().cast(),
        );
        gl.DrawArrays(ffi::TRIANGLES, 0, 6);
        gl.DisableVertexAttribArray(pipeline.encode_prog.attrib_vert as u32);

        let mut read_fbo = 0u32;
        gl.GenFramebuffers(1, &mut read_fbo);
        gl.BindFramebuffer(ffi::READ_FRAMEBUFFER, read_fbo);
        gl.FramebufferTexture2D(
            ffi::READ_FRAMEBUFFER,
            ffi::COLOR_ATTACHMENT0,
            ffi::TEXTURE_2D,
            tex[3].tex_id(),
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
            0,
            0,
            bbw,
            bbh,
            bbx,
            bby,
            bbx + bbw,
            bby + bbh,
            ffi::COLOR_BUFFER_BIT,
            ffi::LINEAR,
        );

        gl.DeleteFramebuffers(1, &mut read_fbo);
        gl.DeleteFramebuffers(1, &mut fbo);
        gl.BindFramebuffer(ffi::DRAW_FRAMEBUFFER, 0);

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

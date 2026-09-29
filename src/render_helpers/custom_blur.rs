use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::{ensure, Context as _};
use niri_config::ShaderPipelineStage;

#[derive(Debug, Clone)]
pub struct FieldPassStep {
    pub name: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub enum RenderPassStep {
    DualKawaseBlur {
        passes: Option<u8>,
        offset: Option<f32>,
    },
    Custom {
        name: String,
        source: String,
        scale: f32,
    },
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub field_passes: Vec<FieldPassStep>,
    pub render_passes: Vec<RenderPassStep>,
}

impl PipelineConfig {
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        1u8.hash(&mut hasher);
        for field in &self.field_passes {
            field.source.hash(&mut hasher);
        }
        for pass in &self.render_passes {
            match pass {
                RenderPassStep::DualKawaseBlur { passes, offset } => {
                    0u8.hash(&mut hasher);
                    passes.hash(&mut hasher);
                    offset.map(f32::to_bits).hash(&mut hasher);
                }
                RenderPassStep::Custom { source, scale, .. } => {
                    1u8.hash(&mut hasher);
                    source.hash(&mut hasher);
                    scale.to_bits().hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }
}

fn read_shader_file(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

pub fn resolve_pipeline(pipeline: &niri_config::ShaderPipeline) -> anyhow::Result<PipelineConfig> {
    ensure!(
        pipeline.version == 1,
        "unsupported shader-pipeline version {}; expected 1",
        pipeline.version
    );

    let mut field_passes = Vec::new();
    let mut render_passes = Vec::new();
    let mut render_started = false;

    for (i, stage) in pipeline.stages.iter().enumerate() {
        match stage {
            ShaderPipelineStage::FieldShader(shader) => {
                ensure!(
                    !render_started,
                    "field-shader at stage {i} must precede every blur and shader stage"
                );
                field_passes.push(FieldPassStep {
                    name: shader.file.clone(),
                    source: read_shader_file(Path::new(&shader.file))?,
                });
            }
            ShaderPipelineStage::Blur(blur) => {
                render_started = true;
                render_passes.push(RenderPassStep::DualKawaseBlur {
                    passes: blur.passes,
                    offset: blur.offset.map(|offset| offset.0 as f32),
                });
            }
            ShaderPipelineStage::Shader(shader) => {
                ensure!(
                    shader.scale.is_finite() && shader.scale > 0.0,
                    "shader at stage {i}: scale must be finite and positive"
                );
                render_started = true;
                render_passes.push(RenderPassStep::Custom {
                    name: shader.file.clone(),
                    source: read_shader_file(Path::new(&shader.file))?,
                    scale: shader.scale,
                });
            }
        }
    }

    ensure!(
        !render_passes.is_empty(),
        "pipeline must have at least one blur or shader stage"
    );

    Ok(PipelineConfig {
        field_passes,
        render_passes,
    })
}

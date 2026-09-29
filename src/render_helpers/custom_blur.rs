use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::ensure;
use niri_config::{ShaderPipelineStage, ShaderSource};

#[derive(Debug, Clone)]
pub enum RenderPassStep {
    DualKawaseBlur {
        passes: Option<u8>,
        offset: Option<f32>,
    },
    Custom {
        name: String,
        source: ShaderSource,
        scale: f32,
    },
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub render_passes: Vec<RenderPassStep>,
}

pub fn cache_key(pipeline: &niri_config::ShaderPipeline) -> u64 {
    let mut hasher = DefaultHasher::new();
    pipeline.version.hash(&mut hasher);
    for stage in &pipeline.stages {
        match stage {
            ShaderPipelineStage::Blur(blur) => {
                0u8.hash(&mut hasher);
                blur.passes.hash(&mut hasher);
                blur.offset
                    .map(|offset| offset.0.to_bits())
                    .hash(&mut hasher);
            }
            ShaderPipelineStage::Shader(shader) => {
                1u8.hash(&mut hasher);
                shader.source.fingerprint().hash(&mut hasher);
                shader.scale.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

pub fn resolve_pipeline(pipeline: &niri_config::ShaderPipeline) -> anyhow::Result<PipelineConfig> {
    ensure!(
        pipeline.version == 1,
        "unsupported shader-pipeline version {}; expected 1",
        pipeline.version
    );

    let mut render_passes = Vec::new();

    for (i, stage) in pipeline.stages.iter().enumerate() {
        match stage {
            ShaderPipelineStage::Blur(blur) => {
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
                render_passes.push(RenderPassStep::Custom {
                    name: shader.source.name(),
                    source: shader.source.clone(),
                    scale: shader.scale,
                });
            }
        }
    }

    ensure!(
        !render_passes.is_empty(),
        "pipeline must have at least one blur or shader stage"
    );

    Ok(PipelineConfig { render_passes })
}

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::{ensure, Context as _};
use niri_config::{MaskPassKind, RenderPassKind};

#[derive(Debug, Clone)]
pub enum MaskPassStep {
    WindowVectors,
    RegionVectors,
    Custom {
        name: String,
        source: String,
        scale: f32,
    },
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
    pub mask_passes: Vec<MaskPassStep>,
    pub render_passes: Vec<RenderPassStep>,
}

impl PipelineConfig {
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for mask in &self.mask_passes {
            match mask {
                MaskPassStep::WindowVectors => {
                    0u8.hash(&mut hasher);
                }
                MaskPassStep::RegionVectors => {
                    1u8.hash(&mut hasher);
                }
                MaskPassStep::Custom { source, scale, .. } => {
                    2u8.hash(&mut hasher);
                    source.hash(&mut hasher);
                    scale.to_bits().hash(&mut hasher);
                }
            }
        }
        for pass in &self.render_passes {
            match pass {
                RenderPassStep::DualKawaseBlur { passes, offset } => {
                    0u8.hash(&mut hasher);
                    passes.hash(&mut hasher);
                    offset.map(|o| o.to_bits()).hash(&mut hasher);
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

pub fn resolve_pipeline(
    pipeline: &niri_config::ShaderPipeline,
) -> anyhow::Result<PipelineConfig> {
    let mut mask_passes = Vec::with_capacity(pipeline.mask_passes.len());
    for (i, mask) in pipeline.mask_passes.iter().enumerate() {
        let step = match mask.kind {
            MaskPassKind::WindowVectors => MaskPassStep::WindowVectors,
            MaskPassKind::RegionVectors => MaskPassStep::RegionVectors,
            MaskPassKind::Custom => {
                let file = mask.file.as_deref().with_context(|| {
                    format!("mask-pass {i}: `custom` requires a `file` property")
                })?;
                ensure!(
                    mask.scale > 0.0,
                    "mask-pass {i}: scale must be positive"
                );
                let source = read_shader_file(Path::new(file))?;
                MaskPassStep::Custom {
                    name: file.to_string(),
                    source,
                    scale: mask.scale,
                }
            }
        };
        mask_passes.push(step);
    }

    ensure!(
        !pipeline.render_passes.is_empty(),
        "pipeline must have at least one render-pass"
    );

    let mut render_passes = Vec::with_capacity(pipeline.render_passes.len());
    for (i, pass) in pipeline.render_passes.iter().enumerate() {
        let step = match pass.kind {
            RenderPassKind::DualKawaseBlur => RenderPassStep::DualKawaseBlur {
                passes: pass.passes,
                offset: pass.offset.map(|o| o.0 as f32),
            },
            RenderPassKind::Custom => {
                let file = pass.file.as_deref().with_context(|| {
                    format!("render-pass {i}: `custom` requires a `file` property")
                })?;
                ensure!(
                    pass.scale > 0.0,
                    "render-pass {i}: scale must be positive"
                );
                let source = read_shader_file(Path::new(file))?;
                RenderPassStep::Custom {
                    name: file.to_string(),
                    source,
                    scale: pass.scale,
                }
            }
        };
        render_passes.push(step);
    }

    Ok(PipelineConfig {
        mask_passes,
        render_passes,
    })
}

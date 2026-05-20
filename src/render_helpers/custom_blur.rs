use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::{ensure, Context as _};

#[derive(Debug, Clone)]
pub struct CustomMaskPassConfig {
    pub name: String,
    pub source: String,
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct CustomRenderPassConfig {
    pub name: String,
    pub source: String,
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub mask_pass: Option<CustomMaskPassConfig>,
    pub render_passes: Vec<CustomRenderPassConfig>,
}

impl PipelineConfig {
    pub fn cache_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        if let Some(mask) = &self.mask_pass {
            mask.source.hash(&mut hasher);
            mask.scale.to_bits().hash(&mut hasher);
        }
        for pass in &self.render_passes {
            pass.source.hash(&mut hasher);
            pass.scale.to_bits().hash(&mut hasher);
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
    let mask_pass = if let Some(mask) = &pipeline.mask_pass {
        ensure!(
            mask.scale > 0.0,
            "mask-pass ({:?}): scale must be positive",
            mask.name
        );
        let source = read_shader_file(Path::new(&mask.file))?;
        Some(CustomMaskPassConfig {
            name: mask.name.clone(),
            source,
            scale: mask.scale,
        })
    } else {
        None
    };

    ensure!(
        !pipeline.render_passes.is_empty(),
        "pipeline must have at least one render-pass"
    );

    let mut render_passes = Vec::with_capacity(pipeline.render_passes.len());
    for (i, pass) in pipeline.render_passes.iter().enumerate() {
        ensure!(
            pass.scale > 0.0,
            "render-pass {} ({:?}): scale must be positive",
            i,
            pass.name
        );
        let source = read_shader_file(Path::new(&pass.file))?;
        render_passes.push(CustomRenderPassConfig {
            name: pass.name.clone(),
            source,
            scale: pass.scale,
        });
    }

    Ok(PipelineConfig {
        mask_pass,
        render_passes,
    })
}



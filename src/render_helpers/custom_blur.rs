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

#[derive(knuffel::Decode, Debug)]
struct RenderPass {
    #[knuffel(argument)]
    name: String,
    #[knuffel(property)]
    file: String,
    #[knuffel(property, default = 1.0)]
    scale: f32,
}

#[derive(knuffel::Decode, Debug)]
struct MaskPass {
    #[knuffel(argument)]
    name: String,
    #[knuffel(property)]
    file: String,
    #[knuffel(property, default = 1.0)]
    scale: f32,
}

#[derive(knuffel::Decode, Debug)]
struct PipelineManifest {
    #[knuffel(child)]
    mask_pass: Option<MaskPass>,
    #[knuffel(children(name = "render-pass"))]
    render_passes: Vec<RenderPass>,
}

pub fn load_custom_blur_pipeline(dir: &Path) -> anyhow::Result<PipelineConfig> {
    ensure!(
        dir.is_dir(),
        "custom shader path is not a directory: {}",
        dir.display()
    );

    let manifest_path = dir.join("pipeline.kdl");
    let manifest_text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;

    let manifest: PipelineManifest =
        knuffel::parse(&manifest_path.to_string_lossy(), &manifest_text)
            .with_context(|| format!("failed to parse {}", manifest_path.display()))?;

    ensure!(
        !manifest.render_passes.is_empty(),
        "pipeline must have at least one render-pass"
    );

    let mask_pass = if let Some(mask) = manifest.mask_pass {
        ensure!(
            mask.scale > 0.0,
            "mask-pass ({:?}): scale must be positive",
            mask.name
        );
        let frag_path = dir.join(&mask.file);
        let source = std::fs::read_to_string(&frag_path).with_context(|| {
            format!(
                "failed to read mask-pass ({:?}): {}",
                mask.name,
                frag_path.display()
            )
        })?;
        Some(CustomMaskPassConfig {
            name: mask.name,
            source,
            scale: mask.scale,
        })
    } else {
        None
    };

    let mut render_passes = Vec::with_capacity(manifest.render_passes.len());
    for (i, pass) in manifest.render_passes.iter().enumerate() {
        ensure!(
            pass.scale > 0.0,
            "render-pass {} ({:?}): scale must be positive",
            i,
            pass.name
        );

        let frag_path = dir.join(&pass.file);
        let source = std::fs::read_to_string(&frag_path).with_context(|| {
            format!(
                "failed to read render-pass {} ({:?}): {}",
                i,
                pass.name,
                frag_path.display()
            )
        })?;

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

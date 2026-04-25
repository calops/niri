use std::path::Path;

use anyhow::{Context as _, ensure};

#[derive(Debug, Clone)]
pub struct CustomBlurPassConfig {
    pub name: String,
    pub source: String,
    pub scale: f32,
}

#[derive(knuffel::Decode, Debug)]
struct PipelinePass {
    #[knuffel(argument)]
    name: String,
    #[knuffel(property)]
    file: String,
    #[knuffel(property, default = 1.0)]
    scale: f32,
}

#[derive(knuffel::Decode, Debug)]
struct PipelineManifest {
    #[knuffel(children)]
    passes: Vec<PipelinePass>,
}

pub fn load_custom_blur_pipeline(
    dir: &Path,
) -> anyhow::Result<Vec<CustomBlurPassConfig>> {
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
        !manifest.passes.is_empty(),
        "pipeline must have at least one pass"
    );

    let mut passes = Vec::with_capacity(manifest.passes.len());
    for (i, pass) in manifest.passes.iter().enumerate() {
        ensure!(
            pass.scale > 0.0,
            "pass {} ({:?}): scale must be positive",
            i,
            pass.name
        );

        let frag_path = dir.join(&pass.file);
        let source = std::fs::read_to_string(&frag_path).with_context(|| {
            format!(
                "failed to read pass {} ({:?}): {}",
                i,
                pass.name,
                frag_path.display()
            )
        })?;

        passes.push(CustomBlurPassConfig {
            name: pass.name.clone(),
            source,
            scale: pass.scale,
        });
    }

    Ok(passes)
}

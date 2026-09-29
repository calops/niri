use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use knuffel::errors::DecodeError;

use crate::{BasePath, Includes};

#[derive(Debug, Clone, Copy)]
pub(crate) enum PositionalShaderSource {
    Inline,
    File,
}

#[derive(Clone, PartialEq)]
pub struct ShaderSource {
    origin: ShaderSourceOrigin,
    source: Arc<str>,
    fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum ShaderSourceOrigin {
    Inline,
    File(PathBuf),
}

impl fmt::Debug for ShaderSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.origin {
            ShaderSourceOrigin::Inline => f.debug_tuple("Inline").field(&self.source).finish(),
            ShaderSourceOrigin::File(path) => f.debug_tuple("File").field(path).finish(),
        }
    }
}

impl ShaderSource {
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn name(&self) -> String {
        match &self.origin {
            ShaderSourceOrigin::Inline => "inline shader".to_owned(),
            ShaderSourceOrigin::File(path) => path.display().to_string(),
        }
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    fn inline(source: String) -> Self {
        Self::new(ShaderSourceOrigin::Inline, source)
    }

    fn file(path: PathBuf, source: String) -> Self {
        Self::new(ShaderSourceOrigin::File(path), source)
    }

    fn new(origin: ShaderSourceOrigin, source: String) -> Self {
        let mut hasher = DefaultHasher::new();
        1u8.hash(&mut hasher);
        match &origin {
            ShaderSourceOrigin::Inline => 0u8.hash(&mut hasher),
            ShaderSourceOrigin::File(path) => {
                1u8.hash(&mut hasher);
                path.hash(&mut hasher);
            }
        }
        source.hash(&mut hasher);
        let fingerprint = hasher.finish();

        Self {
            origin,
            source: source.into(),
            fingerprint,
        }
    }
}

pub(crate) fn decode_shader_source<S>(
    node: &knuffel::ast::SpannedNode<S>,
    ctx: &mut knuffel::decode::Context<S>,
    positional: PositionalShaderSource,
    allowed_properties: &[&str],
) -> Result<ShaderSource, DecodeError<S>>
where
    S: knuffel::traits::ErrorSpan,
{
    if let Some(type_name) = &node.type_name {
        ctx.emit_error(DecodeError::unexpected(
            type_name,
            "type name",
            "no type name expected for this node",
        ));
    }

    let mut argument = None;
    let mut arguments = node.arguments.iter();
    if let Some(value) = arguments.next() {
        argument = Some(knuffel::traits::DecodeScalar::decode(value, ctx)?);
    }
    if let Some(value) = arguments.next() {
        ctx.emit_error(DecodeError::unexpected(
            &value.literal,
            "argument",
            "unexpected argument",
        ));
    }

    let mut inline = None;
    let mut file = None;
    for (name, value) in &node.properties {
        match &***name {
            "inline" => inline = Some(knuffel::traits::DecodeScalar::decode(value, ctx)?),
            "file" => file = Some(knuffel::traits::DecodeScalar::decode(value, ctx)?),
            property if allowed_properties.contains(&property) => {}
            property => ctx.emit_error(DecodeError::unexpected(
                name,
                "property",
                format!("unexpected property `{}`", property.escape_default()),
            )),
        }
    }

    for child in node.children() {
        ctx.emit_error(DecodeError::unexpected(
            child,
            "node",
            format!("unexpected node `{}`", child.node_name.escape_default()),
        ));
    }

    let source_count = usize::from(argument.is_some())
        + usize::from(inline.is_some())
        + usize::from(file.is_some());
    if source_count == 0 {
        return Err(DecodeError::missing(
            node,
            "a shader source argument, `inline=`, or `file=` is required",
        ));
    }
    if source_count > 1 {
        return Err(DecodeError::conversion(
            node,
            "shader source argument, `inline=`, and `file=` are mutually exclusive",
        ));
    }

    if let Some(source) = inline {
        return Ok(ShaderSource::inline(source));
    }

    let file = match (argument, positional) {
        (Some(source), PositionalShaderSource::Inline) => {
            return Ok(ShaderSource::inline(source));
        }
        (Some(file), PositionalShaderSource::File) => file,
        (None, _) => file.unwrap(),
    };

    let path = if let Ok(rest) = PathBuf::from(&file).strip_prefix("~") {
        let Some(home) = std::env::home_dir() else {
            return Err(DecodeError::conversion(
                node,
                format!("error retrieving home directory to expand {file:?}"),
            ));
        };
        home.join(rest)
    } else {
        let base = ctx.get::<BasePath>().unwrap();
        base.0.join(file)
    };

    // Store the dependency even when the read fails so creating or fixing the
    // file triggers another config reload.
    let includes = ctx.get::<Rc<std::cell::RefCell<Includes>>>().unwrap();
    includes.borrow_mut().0.push(path.clone());

    let source = std::fs::read_to_string(&path).map_err(|err| {
        DecodeError::conversion(
            node,
            format!("failed to read shader from {}: {err}", path.display()),
        )
    })?;

    Ok(ShaderSource::file(path, source))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::ShaderSourceOrigin;
    use crate::Config;

    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("niri-config-shader-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn positional_animation_shader_remains_inline() {
        let config = Config::parse_mem(
            r##"
            animations {
                window-open {
                    custom-shader "vec4 open_color() { return vec4(1.0); }"
                }
            }
            "##,
        )
        .unwrap();

        let source = config.animations.window_open.custom_shader.unwrap();
        assert_eq!(source.source(), "vec4 open_color() { return vec4(1.0); }");
        assert_eq!(source.origin, ShaderSourceOrigin::Inline);
    }

    #[test]
    fn shader_files_are_relative_to_the_declaring_include_and_watched() {
        let temp = TempDir::new();
        let effects = temp.0.join("effects");
        std::fs::create_dir_all(&effects).unwrap();

        let include = effects.join("effects.kdl");
        let animation = effects.join("open.frag");
        let color = effects.join("glass.frag");
        std::fs::write(
            &include,
            r##"
            animations {
                window-open {
                    custom-shader file="open.frag"
                }
            }
            blur {
                shader-pipeline version=1 {
                    shader "glass.frag"
                    shader file="glass.frag"
                }
            }
            "##,
        )
        .unwrap();
        std::fs::write(&animation, "animation source").unwrap();
        std::fs::write(&color, "color source").unwrap();

        let result = Config::parse(
            &temp.0.join("config.kdl"),
            r#"include "effects/effects.kdl""#,
        );
        let config = result.config.unwrap();

        assert_eq!(
            config
                .animations
                .window_open
                .custom_shader
                .unwrap()
                .source(),
            "animation source"
        );
        let pipeline = config.blur.shader_pipeline.unwrap();
        assert_eq!(pipeline.stages.len(), 2);
        for stage in &pipeline.stages {
            let crate::ShaderPipelineStage::Shader(shader) = stage else {
                panic!("expected custom shader stage");
            };
            assert_eq!(shader.source.source(), "color source");
        }
        assert!(result.includes.contains(&include));
        assert!(result.includes.contains(&animation));
        assert!(result.includes.contains(&color));
    }
}

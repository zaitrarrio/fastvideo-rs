//! Relative paths inside a model tree, as `model.toml` components and
//! manifest file entries name them: `/`-separated, relative, no `.` or `..`
//! components, no empty components, no `\`, no NUL.

/// Why a string is not a valid in-tree path.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum PathError {
    #[error("empty path")]
    Empty,
    #[error("absolute path (leading `/`)")]
    Absolute,
    #[error("`\\` in path: separators are `/`")]
    Backslash,
    #[error("`..` component")]
    DotDot,
    #[error("`.` component")]
    CurDir,
    #[error("empty component (`//` or a trailing `/`)")]
    EmptyComponent,
    #[error("control character in path")]
    Control,
}

/// Check one in-tree path.
pub fn check_relative_path(path: &str) -> Result<(), PathError> {
    if path.is_empty() {
        return Err(PathError::Empty);
    }
    if path.starts_with('/') {
        return Err(PathError::Absolute);
    }
    if path.contains('\\') {
        return Err(PathError::Backslash);
    }
    if path.chars().any(|c| (c as u32) < 0x20 || c == '\u{7f}') {
        return Err(PathError::Control);
    }
    for component in path.split('/') {
        match component {
            "" => return Err(PathError::EmptyComponent),
            "." => return Err(PathError::CurDir),
            ".." => return Err(PathError::DotDot),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_and_rejects() {
        assert_eq!(
            check_relative_path("weights/transformer.safetensors"),
            Ok(())
        );
        assert_eq!(check_relative_path("model.toml"), Ok(()));
        assert_eq!(check_relative_path(""), Err(PathError::Empty));
        assert_eq!(check_relative_path("/x"), Err(PathError::Absolute));
        assert_eq!(check_relative_path("a\\b"), Err(PathError::Backslash));
        assert_eq!(check_relative_path("a/../b"), Err(PathError::DotDot));
        assert_eq!(check_relative_path("./a"), Err(PathError::CurDir));
        assert_eq!(check_relative_path("a//b"), Err(PathError::EmptyComponent));
        assert_eq!(check_relative_path("a/"), Err(PathError::EmptyComponent));
        assert_eq!(check_relative_path("a\nb"), Err(PathError::Control));
    }
}

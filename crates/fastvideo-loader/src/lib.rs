//! Weight loading. Phase 0 maps names; Phase 1 pulls safetensors via hf-hub.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("weight download/load is not implemented yet: {0}")]
    NotImplemented(String),
}

/// Apply FastVideo `param_names_mapping` rules. Regex rewrite lands in Phase 1;
/// this helper documents the table and returns the source key unchanged until
/// the regex engine is wired.
pub fn map_param_name(source: &str) -> String {
    let _ = fastvideo_models::wan::PARAM_NAMES_MAPPING;
    source.to_string()
}

pub fn load_diffusers_repo(_model_id: &str) -> Result<(), LoaderError> {
    Err(LoaderError::NotImplemented(
        "hf-hub + safetensors download is Phase 1".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_table_is_nonempty() {
        assert!(!fastvideo_models::wan::PARAM_NAMES_MAPPING.is_empty());
        assert_eq!(map_param_name("a.b"), "a.b");
    }
}

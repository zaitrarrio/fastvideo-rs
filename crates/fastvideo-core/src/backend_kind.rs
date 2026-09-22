use std::str::FromStr;

/// Runtime backend. Only cudarc remains; other names are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendKind {
    #[default]
    Cudarc,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        "cudarc"
    }
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cudarc" | "cdarc" | "cuda-native" | "cuda" => Ok(Self::Cudarc),
            "candle" | "luminal" | "luminar" | "host" | "cpu" | "ref" | "burn" => Err(format!(
                "backend `{s}` was removed; use cudarc"
            )),
            other => Err(format!("unknown backend `{other}` (expected cudarc)")),
        }
    }
}

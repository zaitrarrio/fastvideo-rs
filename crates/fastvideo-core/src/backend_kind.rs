use std::str::FromStr;

/// Runtime backend selected by the user. Maps onto a `TensorBackend` impl.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Host,
    Burn,
    Candle,
    Luminal,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Burn => "burn",
            Self::Candle => "candle",
            Self::Luminal => "luminal",
        }
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
            "host" | "cpu" | "ref" => Ok(Self::Host),
            "burn" => Ok(Self::Burn),
            "candle" => Ok(Self::Candle),
            "luminal" | "luminar" => Ok(Self::Luminal),
            other => Err(format!(
                "unknown backend `{other}` (expected host, burn, candle, or luminal)"
            )),
        }
    }
}

//! Camera trajectory ids for GEN3C (MoGe warp path). Spec: docs/ports/gen3c.md.

/// FastVideo / NVIDIA default trajectory presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryType {
    Left,
    Right,
    ZoomIn,
    ZoomOut,
    Clockwise,
    Counterclockwise,
}

impl TrajectoryType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
            Self::ZoomIn => "zoom_in",
            Self::ZoomOut => "zoom_out",
            Self::Clockwise => "clockwise",
            Self::Counterclockwise => "counterclockwise",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "left" => Some(Self::Left),
            "right" => Some(Self::Right),
            "zoom_in" | "zoomin" => Some(Self::ZoomIn),
            "zoom_out" | "zoomout" => Some(Self::ZoomOut),
            "clockwise" | "cw" => Some(Self::Clockwise),
            "counterclockwise" | "counter_clockwise" | "ccw" => Some(Self::Counterclockwise),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraRotation {
    CenterFacing,
    NoRotation,
}

impl CameraRotation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CenterFacing => "center_facing",
            Self::NoRotation => "no_rotation",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "center_facing" | "center-facing" => Some(Self::CenterFacing),
            "no_rotation" | "none" => Some(Self::NoRotation),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_defaults() {
        assert_eq!(TrajectoryType::parse("left"), Some(TrajectoryType::Left));
        assert_eq!(
            TrajectoryType::parse("ZOOM_IN"),
            Some(TrajectoryType::ZoomIn)
        );
        assert_eq!(
            CameraRotation::parse("center_facing"),
            Some(CameraRotation::CenterFacing)
        );
    }
}

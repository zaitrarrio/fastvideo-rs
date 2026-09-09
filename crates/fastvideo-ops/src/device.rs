/// Logical device. Concrete backends map this onto CPU / CUDA / Metal / WGPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda { index: usize },
    Metal { index: usize },
    Wgpu { index: usize },
}

impl Device {
    pub fn cpu() -> Self {
        Self::Cpu
    }
}

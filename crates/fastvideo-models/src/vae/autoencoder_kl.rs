//! Diffusers `AutoencoderKL` host config (2D image VAE).

#[derive(Debug, Clone, PartialEq)]
pub struct AutoencoderKlConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f32,
    pub spatial_compression_ratio: usize,
}

impl AutoencoderKlConfig {
    /// Classic SD 1.x / 2.x (4 latent channels).
    pub fn sd_legacy() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.18215,
            spatial_compression_ratio: 8,
        }
    }

    /// SD3 / Z-Image / GLM-Image style 16-ch AutoencoderKL.
    pub fn sd3() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 1.5305,
            spatial_compression_ratio: 8,
        }
    }

    /// FLUX.1 AutoencoderKL (16 latent ch; DiT sees packed 64).
    pub fn flux() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.3611,
            spatial_compression_ratio: 8,
        }
    }

    /// FLUX.2 AutoencoderKL (16 latent ch; DiT often sees packed 128).
    pub fn flux2() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.3611,
            spatial_compression_ratio: 8,
        }
    }

    pub fn tiny(latent_channels: usize) -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels,
            block_out_channels: vec![32, 64],
            layers_per_block: 1,
            scaling_factor: 1.0,
            spatial_compression_ratio: 8,
        }
    }

    pub fn latent_spatial(&self, height: usize, width: usize) -> (usize, usize) {
        (
            height / self.spatial_compression_ratio,
            width / self.spatial_compression_ratio,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd3_latent_grid() {
        let c = AutoencoderKlConfig::sd3();
        assert_eq!(c.latent_spatial(1024, 1024), (128, 128));
        assert_eq!(c.latent_channels, 16);
    }
}

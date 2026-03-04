use encase::ShaderType;

#[derive(Debug, Copy, Clone, ShaderType)]
pub struct FilterParams {
    pub kernel_radius: f32,
    pub sharpness: f32,
    pub q_value: f32,
    pub eccentricity: f32,
    pub num_sectors: u32,
    pub blur_kernel_size: u32,
    pub blur_sigma: f32,
    pub dithering: u32,
    pub dithering_str: f32,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            kernel_radius: 8.0,
            sharpness: 8.0,
            q_value: 4.0,
            eccentricity: 1.0,
            num_sectors: 8,
            blur_kernel_size: 6,
            blur_sigma: 3.0,
            dithering: 0,
            dithering_str: 0.5,
        }
    }
}

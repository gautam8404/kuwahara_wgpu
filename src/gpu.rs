use crate::params::FilterParams;
use anyhow::{anyhow, Result};
use eframe::egui::{Image, TextureId, Vec2};
use eframe::wgpu::TextureViewDescriptor;
use eframe::wgpu::{self, TextureUsages};
use encase::{ShaderType, UniformBuffer};
use futures;
use image;
use wgpu::include_wgsl;

pub struct GpuProcessor {
    input_texture: wgpu::Texture,
    pub output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    input_view: wgpu::TextureView,
    sobel_texture: wgpu::Texture,
    blur_texture: wgpu::Texture,

    sobel_pipeline: wgpu::ComputePipeline,
    sobel_bind_group: wgpu::BindGroup,

    blur_pipeline: wgpu::ComputePipeline,
    blur_bind_group: wgpu::BindGroup,

    // multiple pipelines for multiple passes
    // sobel -> gaussian blur -> kuwahara
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,

    pub input_egui_texture_id: eframe::egui::TextureId,
    pub egui_texture_id: eframe::egui::TextureId,
    pub image_width: u32,
    pub image_height: u32,
}

impl GpuProcessor {
    pub fn new(
        render_state: &eframe::egui_wgpu::RenderState,
        image_bytes: &[u8],
        filter_params: FilterParams,
    ) -> Result<Self> {
        let img = image::load_from_memory(image_bytes)?;
        let rgba = img.to_rgba8();
        let raw_rgba = rgba.as_raw();
        let dimensions = rgba.dimensions();
        let (width, height) = dimensions.clone();

        let (device, queue) = (&render_state.device, &render_state.queue);

        let texture_size = wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        };

        let input_texture = create_texture(
            &device,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("input_texture"),
            texture_size,
        );

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &raw_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );

        let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let input_texture_id = render_state.renderer.write().register_native_texture(
            device,
            &input_view,
            wgpu::FilterMode::Linear,
        );

        let sobel_storage_texture = create_texture(
            &device,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            wgpu::TextureFormat::Rgba16Float,
            Some("sobel_texture"),
            texture_size,
        );

        let sobel_storage_view =
            sobel_storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blur_storage_texture = create_texture(
            &device,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            wgpu::TextureFormat::Rgba16Float,
            Some("blur_texture"),
            texture_size,
        );

        let blur_storage_view =
            blur_storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let output_texture = create_texture(
            &device,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("output_texture"),
            texture_size,
        );

        let output_view = output_texture.create_view(&TextureViewDescriptor::default());

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Buffer"),
            size: FilterParams::min_size().get(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sobel_shader = device.create_shader_module(include_wgsl!("../shaders/sobel.wgsl"));
        let (sobel_pipeline, sobel_bind_group, _) = create_compute_pipeline_with_view(
            &device,
            sobel_shader,
            "sobel",
            "main",
            vec![
                Resource::Texture(&input_view),
                Resource::StorageTexture(&sobel_storage_view, wgpu::TextureFormat::Rgba16Float),
            ],
        )?;

        let blur_shader =
            device.create_shader_module(include_wgsl!("../shaders/gaussian_blur.wgsl"));
        let (blur_pipeline, blur_bind_group, _) = create_compute_pipeline_with_view(
            &device,
            blur_shader,
            "blur",
            "main",
            vec![
                Resource::Texture(&sobel_storage_view),
                Resource::StorageTexture(&blur_storage_view, wgpu::TextureFormat::Rgba16Float),
                Resource::UniformBuffer(&buffer),
            ],
        )?;

        let shader = device.create_shader_module(include_wgsl!("../shaders/kuwahara.wgsl"));
        let (pipeline, bind_group, bind_group_layout) = create_compute_pipeline_with_view(
            &device,
            shader,
            "kuwahara",
            "main",
            vec![
                Resource::Texture(&input_view),
                Resource::Texture(&blur_storage_view),
                Resource::StorageTexture(&output_view, wgpu::TextureFormat::Rgba8Unorm),
                Resource::UniformBuffer(&buffer),
            ], // &input_view,
               // &output_view,
        )?;

        let texture_id = render_state.renderer.write().register_native_texture(
            device,
            &output_view,
            wgpu::FilterMode::Linear,
        );

        Ok(Self {
            input_texture,
            output_texture,
            output_view,
            input_view,

            blur_texture: blur_storage_texture,
            blur_pipeline: blur_pipeline,
            blur_bind_group: blur_bind_group,

            sobel_pipeline,
            sobel_texture: sobel_storage_texture,
            sobel_bind_group,

            input_egui_texture_id: input_texture_id,
            egui_texture_id: texture_id,
            image_width: dimensions.0,
            image_height: dimensions.1,
            compute_pipeline: pipeline,
            bind_group: bind_group,
            bind_group_layout: bind_group_layout,
            params_buffer: buffer,
        })
    }

    pub fn run_filter(&self, render_state: &eframe::egui_wgpu::RenderState, params: FilterParams) {
        let (device, queue) = (&render_state.device, &render_state.queue);

        let mut encase_buffer = UniformBuffer::new(Vec::new());
        encase_buffer.write(&params).unwrap();
        let bytes: &[u8] = encase_buffer.as_ref();

        queue.write_buffer(&self.params_buffer, 0, bytes);

        let mut compute_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass =
                compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute pass"),
                    timestamp_writes: None,
                });

            let workgroup_x = (self.image_width + 15) / 16;
            let workgroup_y = (self.image_height + 15) / 16;

            // Sobel Pass
            compute_pass.set_pipeline(&self.sobel_pipeline);
            compute_pass.set_bind_group(0, Some(&self.sobel_bind_group), &[]);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);

            //Blur Pass
            compute_pass.set_pipeline(&self.blur_pipeline);
            compute_pass.set_bind_group(0, Some(&self.blur_bind_group), &[]);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);

            // Kuwahara Pass
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, Some(&self.bind_group), &[]);
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        queue.submit(Some(compute_encoder.finish()));
    }


    pub fn update_egui_texture(&self, render_state: &eframe::egui_wgpu::RenderState) {
        render_state
            .renderer
            .write()
            .update_egui_texture_from_wgpu_texture(
                &render_state.device,
                &self.output_view,
                wgpu::FilterMode::Linear,
                self.egui_texture_id,
            );
    }

    pub fn destroy(&self, render_state: &eframe::egui_wgpu::RenderState) {
        render_state
            .renderer
            .write()
            .free_texture(&self.egui_texture_id);
    }
}

pub fn create_texture(
    device: &wgpu::Device,
    texture_usage: wgpu::TextureUsages,
    texture_format: wgpu::TextureFormat,
    label: Option<&str>,
    size: wgpu::Extent3d,
) -> wgpu::Texture {
    return device.create_texture(&wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture_format,
        usage: texture_usage,
        view_formats: &[],
    });
}

pub enum Resource<'a> {
    Texture(&'a wgpu::TextureView),
    StorageTexture(&'a wgpu::TextureView, wgpu::TextureFormat),
    UniformBuffer(&'a wgpu::Buffer),
}

pub fn create_compute_pipeline_with_view(
    device: &wgpu::Device,
    shader: wgpu::ShaderModule,
    prefix_label: &str,
    entry_point: &str,
    // input_view: &wgpu::TextureView,
    // output_view: &wgpu::TextureView,
    // output_texture_format: &wgpu::TextureFormat
    resources: Vec<Resource>,
) -> Result<(
    wgpu::ComputePipeline,
    wgpu::BindGroup,
    wgpu::BindGroupLayout,
)> {
    // let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    //     label: Some(format!("{}_compute_shader_moudle", prefix_label).as_str()),
    //     source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source.into())),
    // });

    let mut layouts = Vec::with_capacity(resources.len());
    let mut bind_groups = Vec::with_capacity(resources.len());

    for (i, resource) in resources.iter().enumerate() {
        let binding = i as u32;
        match resource {
            Resource::Texture(texture_view) => {
                layouts.push(wgpu::BindGroupLayoutEntry {
                    binding: binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
                bind_groups.push(wgpu::BindGroupEntry {
                    binding: binding,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                });
            }
            Resource::StorageTexture(texture_view, texture_format) => {
                layouts.push(wgpu::BindGroupLayoutEntry {
                    binding: binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: texture_format.to_owned(),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                });
                bind_groups.push(wgpu::BindGroupEntry {
                    binding: binding,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                });
            }
            Resource::UniformBuffer(buffer) => {
                layouts.push(wgpu::BindGroupLayoutEntry {
                    binding: binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                bind_groups.push(wgpu::BindGroupEntry {
                    binding: binding,
                    resource: buffer.as_entire_binding(),
                });
            }
        }
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(format!("{}_bind_group_layout", prefix_label).as_str()),
        entries: layouts.as_slice(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(format!("{}_bind_group", prefix_label).as_str()),
        layout: &bind_group_layout,
        entries: bind_groups.as_slice(),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(format!("{}_compute_shader_moudle", prefix_label).as_str()),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    Ok((
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(format!("{}_compute_shader_moudle", prefix_label).as_str()),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: Default::default(),
        }),
        bind_group,
        bind_group_layout,
    ))
}

pub async fn readback_image(
    image_width: u32,
    image_height: u32,
    output_texture: wgpu::Texture,
    render_state: &eframe::egui_wgpu::RenderState,
) -> Result<Vec<u8>> {
    let (device, queue) = (&render_state.device, &render_state.queue);
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded_bytes_per_row = image_width * 4;
    let padding = (align - (unpadded_bytes_per_row % align)) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padding;
    let buffer_size = (padded_bytes_per_row * image_height) as wgpu::BufferAddress;

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(image_height),
            },
        },
        wgpu::Extent3d {
            width: image_width,
            height: image_height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = readback_buffer.slice(..);

    let (sender, receiver) = futures::channel::oneshot::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });

    let width = image_width;
    let height = image_height;
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let map_result = receiver.await.map_err(|_| anyhow!("Channel cancelled"))?;

    map_result.map_err(|e| anyhow!("GPU error: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * image_height) as usize);

    for chunk in data.chunks(padded_bytes_per_row as usize) {
        pixels.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
    }

    drop(data);
    readback_buffer.unmap();
    Ok(pixels)
}
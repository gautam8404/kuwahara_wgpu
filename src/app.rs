use crate::{gpu::GpuProcessor, params::FilterParams};
use eframe::{egui, egui_wgpu};
use log;
use std::sync::mpsc::{channel, Receiver, Sender};

#[cfg(not(target_arch = "wasm32"))]
use pollster;

pub struct KuwaharaApp {
    render_state: Option<egui_wgpu::RenderState>,
    image_bytes: Option<Vec<u8>>,
    params: FilterParams,
    gpu: Option<GpuProcessor>,
    filter_applied: bool,
    refresh_pass: bool,
    status: Option<String>,
    file_receiver: Receiver<Vec<u8>>,
    file_sender: Sender<Vec<u8>>,
}

impl KuwaharaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc.wgpu_render_state.clone();

        if render_state.is_none() {
            log::warn!("no wgpu render state available eframe might not be using wgpu backend");
        }

        let (file_sender, file_receiver) = channel();

        Self {
            render_state,
            image_bytes: None,
            params: FilterParams::default(),
            gpu: None,
            filter_applied: false,
            refresh_pass: false,
            status: None,
            file_receiver,
            file_sender,
        }
    }

    fn load_image(&mut self, bytes: Vec<u8>, frame: &eframe::Frame) {
        let render_state = match frame.wgpu_render_state() {
            Some(rs) => rs,
            None => {
                log::info!("wgpu not available");
                self.status = Some("GPU not available".to_string());
                return;
            }
        };

        if let Some(old) = self.gpu.take() {
            old.destroy(&render_state);
        }

        match GpuProcessor::new(&render_state, &bytes, self.params) {
            Ok(gpu) => {
                self.status = Some(format!(
                    "Loaded Image: {}x{}",
                    gpu.image_width, gpu.image_height
                ));
                self.gpu = Some(gpu);
                self.image_bytes = Some(bytes);
                self.filter_applied = false;
            }
            Err(e) => {
                // log::error!("Error occured: {:?}", e);
                self.status = Some(format!("Error: {e}"));
            }
        }
    }

    fn pick_file(&mut self, ctx: &egui::Context) {
        let sender = self.file_sender.clone();
        let ctx = ctx.clone();
        self.status = Some("Loading Image".to_string());

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                let rt = pollster::FutureExt::block_on(
                    rfd::AsyncFileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
                        .pick_file(),
                );

                if let Some(file) = rt {
                    let bytes = pollster::FutureExt::block_on(file.read());
                    let _ = sender.send(bytes);
                    ctx.request_repaint();
                }
            });
        }
        // {
        //     if let Some(path) = rfd::FileDialog::new()
        //         .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
        //         .pick_file()
        //     {
        //         match std::fs::read(&path) {
        //             Ok(bytes) => self.load_image(bytes),
        //             Err(e) => {
        //                 self.status = Some(format!("Failed to read file: {e}"));
        //             }
        //         }
        //     }
        // }

        #[cfg(target_arch = "wasm32")]
        {
            // self.status = Some("On web, drag & drop an image onto the window".to_string());
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(path) = rfd::AsyncFileDialog::new()
                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
                    .pick_file()
                    .await
                {
                    log::info!("picked file: {:?}", path);
                    let bytes = path.read().await;
                    sender.send(bytes).expect("Failed to send file bytes");
                    ctx.request_repaint();
                }
            });
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let dropped: Vec<_> = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            if let Some(bytes) = file.bytes {
                self.load_image(bytes.to_vec(), frame);
                return;
            }
            if let Some(path) = &file.path {
                if let Ok(bytes) = std::fs::read(path) {
                    self.load_image(bytes, frame);
                    return;
                }
            }
        }
    }

    fn apply_filter(&mut self, frame: &eframe::Frame) {
        let render_state = match frame.wgpu_render_state() {
            Some(rs) => rs,
            None => return,
        };

        if let Some(processor) = &self.gpu {
            processor.run_filter(render_state, self.params);
            processor.update_egui_texture(render_state);
            self.filter_applied = true;
            self.status = Some("Filter applied!".to_string());
        }
    }

    fn refresh_pass(&mut self, frame: &eframe::Frame) {
        log::info!("Refresh pass, reloading image");

        if self.gpu.is_some() && self.filter_applied {
            self.apply_filter(frame);
        }
        self.refresh_pass = false;
    }
}

impl eframe::App for KuwaharaApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.handle_dropped_files(ctx, frame);
        let mut show_og = false;

        if let Ok(bytes) = self.file_receiver.try_recv() {
            self.load_image(bytes, frame);
        }

        if self.refresh_pass {
            self.refresh_pass(frame);
        }

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(250.)
            .show(ctx, |ui| {
                ui.heading("Kuwahara Filter");
                ui.separator();

                if ui.button("open image").clicked() {
                    self.pick_file(&ctx);
                }

                if let Some(label) = &self.status {
                    ui.label(label.as_str());
                }

                ui.separator();

                ui.heading("Parameters");
                ui.add_space(4.0);

                let mut changed = false;

                ui.horizontal(|ui| {
                    ui.label("Kernel Radius");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.params.kernel_radius,
                            1.0..=16.0,
                        ))
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Sharpness");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.params.sharpness, 6.0..=12.0))
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Eccentricity");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.params.eccentricity, 0.0..=2.0))
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Sectors");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.params.num_sectors, 4..=16))
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Blur Radius");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.params.blur_kernel_size, 4..=16))
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Blur Sigma");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.params.blur_sigma, 4.0..=8.0))
                        .changed();
                });


                ui.horizontal(|ui| {
                    let mut dither_bool = self.params.dithering > 0;

                    if ui.checkbox(&mut dither_bool, "Enable Dithering").changed() {
                        // If the checkbox is clicked, convert the bool back to u32 (1 or 0)
                        self.params.dithering = if dither_bool { 1 } else { 0 };
                        log::info!("dithering = {}", self.params.dithering);
                        changed = true;
                    }
                });

                ui.horizontal(|ui| {
                    changed |= ui
                        .button("Compare")
                        .clicked();
                });



                ui.add_space(8.);
                ui.separator();

                let has_image = self.gpu.is_some();

                if ui
                    .add_enabled(has_image, egui::Button::new("Apply Filter"))
                    .clicked()
                {
                    self.apply_filter(frame);
                }

                if has_image {
                    ui.label(format!("Filter Applied: {}", self.filter_applied));
                }

                if changed {
                    self.refresh_pass = true;
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.gpu.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.available_height() / 3.0);
                        ui.heading("Drop an image here or click Open Image");
                        ui.add_space(8.0);
                        ui.label("Supported: PNG, JPEG, BMP, WebP");
                    });
                });
                return;
            }

            let processor = self.gpu.as_ref().unwrap();
            let mut tex_id = processor.egui_texture_id;
            if !self.filter_applied || show_og {
                // self.show_input_preview(ui);
                tex_id = processor.input_egui_texture_id;
            }

            let img_size =
                egui::Vec2::new(processor.image_width as f32, processor.image_height as f32);

            let avaialble_size = ui.available_size();
            let scale = (avaialble_size.x / img_size.x)
                .min(avaialble_size.y / img_size.y)
                .min(1.0);
            let display_size = img_size * scale;

            egui::ScrollArea::both().show(ui, |ui| {
                ui.add(
                    egui::Image::new(egui::load::SizedTexture::new(tex_id, display_size))
                        .fit_to_original_size(1.0),
                );
            });
        });
    }
}

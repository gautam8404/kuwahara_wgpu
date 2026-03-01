@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> filter_params: FilterParams;

struct FilterParams {
  kernerl_radius: f32,
  sharpness: f32,
  eccentricity: f32,
  num_sectors: u32,
  blur_kernel_size: u32,
  blur_sigma: f32
}


@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
   // just inverse pixels
  let dimensions = vec2<i32>(textureDimensions(input_texture));
  let coords = vec2<i32>(global_id.xy);


  if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
      return;
  }

  let pixel = textureLoad(input_texture, coords, 0);

  let gaussian_weights = array<f32,9>(1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);

  let radius = i32(filter_params.blur_kernel_size);
  let sigma = filter_params.blur_sigma;

  var color = vec3<f32>(0.0);
  var weight_sum = 0.0;

  for (var y: i32 = -radius; y <= radius; y++) {
    for (var x: i32 = -radius; x <= radius; x++) {
        var ncord = coords + vec2<i32>(x,y);
        ncord = clamp(ncord, vec2<i32>(0, 0), dimensions - vec2<i32>(1, 1));
        let ncol = textureLoad(input_texture, ncord, 0).rgb;

        let distance_squared = f32(x * x + y * y);
        let weight = exp(-distance_squared / (2.0 * sigma * sigma));

        color += ncol * weight;
        weight_sum += weight;
    }
  }
  color = color/weight_sum;


  textureStore(output_texture, coords, vec4<f32>(color, pixel.a));

}


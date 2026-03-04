@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_tensor: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: FilterParams;

struct FilterParams {
  kernerl_radius: f32,
  hardness: f32,
  q_value: f32,  
  alpha: f32,
  num_sectors: u32,
  blur_kernel_size: u32,
  blur_sigma: f32,
  dithering: u32,
  dithering_str: f32
}


fn get_dither_noise(pixel_coords: vec2<f32>, col: vec3<f32>) -> vec3<f32> {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    
    let noise = fract(magic.z * fract(dot(pixel_coords, magic.xy)));
    
    let strength = 0.6;
    let centered_noise = (noise - 0.5) * params.dithering_str;
    
    return saturate(col + vec3<f32>(centered_noise));
}


@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = vec2<i32>(textureDimensions(input_texture));
    let coords = vec2<i32>(global_id.xy);

    
    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let pixel = textureLoad(input_texture, coords, 0);
    let tensor = textureLoad(input_tensor, coords, 0).rgb;

    let E = tensor.x;
    let F = tensor.y;
    let G = tensor.z;

    let root = sqrt((E - G) * (E - G) + 4.0 * F * F);
    let lambda1 = 0.5 * (E + G + root);
    let lambda2 = 0.5 * (E + G - root);

    let v_dir = vec2<f32>(lambda1 - E, -F);
    var t = vec2<f32>(0.0, 1.0);
    if (length(v_dir) > 0.0) {
        t = normalize(v_dir);
    }

    let phi = -atan2(t.y, t.x);
    var A = 0.0;
    if (lambda1 + lambda2 > 0.0) {
        A = (lambda1 - lambda2) / (lambda1 + lambda2);
    }

    let alpha = params.alpha;
    let radius = params.kernerl_radius;

    let a = radius * clamp((alpha + A) / alpha, 0.1, 2.0);
    let b = radius * clamp(alpha / (alpha + A), 0.1, 2.0);

    let cos_phi = cos(phi);
    let sin_phi = sin(phi);

    let R = mat2x2<f32>(cos_phi, sin_phi, -sin_phi, cos_phi);
    let S = mat2x2<f32>(0.5 / a, 0.0, 0.0, 0.5 / b);
    let SR = S * R;

    let max_x = i32(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
    let max_y = i32(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));

    let zeta = 2.0/params.kernerl_radius;
    let sinZeroCross = sin(0.392699);
    let eta = (zeta + cos(0.392699)) / (sinZeroCross * sinZeroCross);

    // Arrays to hold sector sums
    var m: array<vec4<f32>, 8>; // Color sum
    var s: array<vec3<f32>, 8>; // Variance sum

    // 4. THE 2D SAMPLING LOOP
    for (var y: i32 = -max_y; y <= max_y; y++) {
        for (var x: i32 = -max_x; x <= max_x; x++) {

            var v = SR * vec2<f32>(f32(x), f32(y));

            if (dot(v, v) <= 0.25) {
                var sample_coord = coords + vec2<i32>(x, y);
                sample_coord = clamp(sample_coord, vec2<i32>(0, 0), dimensions - vec2<i32>(1, 1));
                let c = clamp(textureLoad(input_texture, sample_coord, 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));

                var sum = 0.0;
                var w :array<f32, 8>;
                var z: f32; var vxx: f32; var vyy: f32;

                // Polynomial Weights Math
                vxx = zeta - eta * v.x * v.x;
                vyy = zeta - eta * v.y * v.y;

                z = max(0.0, v.y + vxx); w[0] = z * z; sum += w[0];
                z = max(0.0, -v.x + vyy); w[2] = z * z; sum += w[2];
                z = max(0.0, -v.y + vxx); w[4] = z * z; sum += w[4];
                z = max(0.0, v.x + vyy); w[6] = z * z; sum += w[6];

                v = sqrt(2.0) / 2.0 * vec2<f32>(v.x - v.y, v.x + v.y);
                vxx = zeta - eta * v.x * v.x;
                vyy = zeta - eta * v.y * v.y;

                z = max(0.0, v.y + vxx); w[1] = z * z; sum += w[1];
                z = max(0.0, -v.x + vyy); w[3] = z * z; sum += w[3];
                z = max(0.0, -v.y + vxx); w[5] = z * z; sum += w[5];
                z = max(0.0, v.x + vyy); w[7] = z * z; sum += w[7];

                let g = exp(-3.125 * dot(v, v)) / sum;

                // Add to sector buckets
                for (var k: i32 = 0; k < 8; k++) {
                    let wk = w[k] * g;
                    m[k] += vec4<f32>(c * wk, wk);
                    s[k] += c * c * wk;
                }
            }
        }
    }

    var final_color = vec4<f32>(0.0);

    for (var k: i32 = 0; k < 8; k++) {
        m[k] = vec4<f32>(m[k].rgb / m[k].w, m[k].w);
        s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);

        let sigma2 = s[k].r + s[k].g + s[k].b;
        let w = 1.0 / (1.0 + pow(params.hardness * 1000.0 * sigma2, params.q_value)); 
        final_color += vec4<f32>(m[k].rgb * w, w);
    }

    final_color = clamp(final_color / final_color.w, vec4<f32>(0.0), vec4<f32>(1.0));

    let original_alpha = textureLoad(input_texture, coords, 0).a;

    if params.dithering > 0 {
        let noise = get_dither_noise(vec2<f32>(coords), final_color.rgb);
        final_color = vec4(noise.rgb, 1.0);
    }
    

    textureStore(output_texture, coords, vec4<f32>(final_color.rgb, original_alpha));
}

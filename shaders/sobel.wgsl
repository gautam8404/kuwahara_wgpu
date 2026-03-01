@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;

@compute
@workgroup_size(16,16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = vec2<i32>(textureDimensions(input_texture));
    let coords = vec2<i32>(global_id.xy);

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    var sampleTex: array<vec3<f32>, 9>;
    var k: i32 = 0;

    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            var ncord = coords + vec2<i32>(x,y);
            ncord = clamp(ncord, vec2<i32>(0, 0), dimensions - vec2<i32>(1, 1));
            sampleTex[k] = textureLoad(input_texture, ncord, 0).rgb;
            k++;
        }
    }

    let Gx = array<f32, 9>(
        -1.0,  0.0,  1.0,
        -2.0,  0.0,  2.0,
        -1.0,  0.0,  1.0
    );

    let Gy = array<f32, 9>(
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    );

    var edgeX = vec3<f32>(0.0);
    var edgeY = vec3<f32>(0.0);

    for (var i: i32; i < 9; i++) {
        edgeX += sampleTex[i] * Gx[i];
        edgeY += sampleTex[i] * Gy[i];
    }

    let E = dot(edgeX ,edgeX);
    let F = dot(edgeX , edgeY);
    let G = dot(edgeY , edgeY);

    textureStore(output_texture, coords, vec4<f32>(E, F , G, 1.0));
}

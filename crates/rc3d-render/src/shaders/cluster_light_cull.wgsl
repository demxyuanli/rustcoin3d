// Clustered forward light culling
// Builds a 3D cluster grid (X×Y×Z) and assigns lights to clusters.

const TILE_SIZE: u32 = 64u;
const CLUSTERS_X: u32 = 16u;
const CLUSTERS_Y: u32 = 8u;
const CLUSTERS_Z: u32 = 24u;
const MAX_LIGHTS_PER_CLUSTER: u32 = 64u;

struct PointLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
};

struct SpotLight {
    position: vec3<f32>,
    direction: vec3<f32>,
    radius: f32,
    cos_inner: f32,
    cos_outer: f32,
    color: vec3<f32>,
    intensity: f32,
};

struct LightCullParams {
    inv_proj: mat4x4<f32>,
    screen_width: u32,
    screen_height: u32,
    num_point_lights: u32,
    num_spot_lights: u32,
    z_near: f32,
    z_far: f32,
    depth_slice_scale: f32,
    depth_slice_bias: f32,
};

@group(0) @binding(0) var<uniform> params: LightCullParams;
@group(0) @binding(1) var<storage, read> point_lights: array<PointLight>;
@group(0) @binding(2) var<storage, read> spot_lights: array<SpotLight>;
@group(0) @binding(3) var<storage, read_write> light_grid: array<u32>;
@group(0) @binding(4) var<storage, read_write> light_index_list: array<u32>;

fn screen_to_view(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view = params.inv_proj * clip;
    return view.xyz / view.w;
}

fn linear_depth_to_slice(linear_depth: f32) -> u32 {
    let z_eye = params.z_near + linear_depth * (params.z_far - params.z_near);
    let slice_f = params.depth_slice_scale * log2(z_eye / params.z_near + 1.0) + params.depth_slice_bias;
    return u32(clamp(slice_f, 0.0, f32(CLUSTERS_Z - 1u)));
}

fn sphere_intersects_frustum_cluster(
    center: vec3<f32>,
    radius: f32,
    cluster_uv_min: vec2<f32>,
    cluster_uv_max: vec2<f32>,
    depth_near: f32,
    depth_far: f32,
) -> bool {
    // Test against cluster near/far depth planes
    if center.z + radius < depth_near || center.z - radius > depth_far {
        return false;
    }

    // Project sphere center to screen and test against UV bounds
    let proj_center_x = center.x / (params.z_near + depth_near);
    let proj_center_y = center.y / (params.z_near + depth_near);
    let proj_radius = radius / (params.z_near + depth_near) * 2.0;

    if proj_center_x + proj_radius < cluster_uv_min.x || proj_center_x - proj_radius > cluster_uv_max.x {
        return false;
    }
    if proj_center_y + proj_radius < cluster_uv_min.y || proj_center_y - proj_radius > cluster_uv_max.y {
        return false;
    }

    return true;
}

fn cone_intersects_frustum_cluster(
    position: vec3<f32>,
    direction: vec3<f32>,
    radius: f32,
    cos_angle: f32,
    cluster_uv_min: vec2<f32>,
    cluster_uv_max: vec2<f32>,
    depth_near: f32,
    depth_far: f32,
) -> bool {
    // Simplified: treat spot light as bounding sphere for cluster test
    let effective_radius = radius * (1.0 / max(cos_angle, 0.01));
    return sphere_intersects_frustum_cluster(
        position,
        effective_radius,
        cluster_uv_min,
        cluster_uv_max,
        depth_near,
        depth_far,
    );
}

@compute @workgroup_size(TILE_SIZE)
fn cluster_light_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    if cx >= CLUSTERS_X { return; }
    let cy = gid.y;
    if cy >= CLUSTERS_Y { return; }

    let cluster_idx = cx + cy * CLUSTERS_X;

    // Compute cluster screen-space bounds
    let u_min = f32(cx) / f32(CLUSTERS_X);
    let u_max = f32(cx + 1u) / f32(CLUSTERS_X);
    let v_min = f32(cy) / f32(CLUSTERS_Y);
    let v_max = f32(cy + 1u) / f32(CLUSTERS_Y);

    // Accumulate light indices for all Z slices of this XY tile
    var light_count: u32 = 0u;
    let grid_base = cluster_idx * CLUSTERS_Z * 2u;

    for (var cz = 0u; cz < CLUSTERS_Z; cz = cz + 1u) {
        let slice_start = grid_base + cz * 2u;

        // Compute depth range for this slice (logarithmic Z distribution)
        let z0 = (params.z_near / params.z_near) - 1.0; // placeholder
        let depth_near = params.z_near * pow(params.z_far / params.z_near, f32(cz) / f32(CLUSTERS_Z));
        let depth_far = params.z_near * pow(params.z_far / params.z_near, f32(cz + 1u) / f32(CLUSTERS_Z));

        var cluster_light_count: u32 = 0u;

        // Test point lights
        for (var i = 0u; i < params.num_point_lights; i = i + 1u) {
            let pl = point_lights[i];
            if sphere_intersects_frustum_cluster(
                pl.position, pl.radius,
                vec2<f32>(u_min, v_min), vec2<f32>(u_max, v_max),
                depth_near, depth_far,
            ) {
                if cluster_light_count < MAX_LIGHTS_PER_CLUSTER {
                    let list_idx = light_count + cluster_light_count;
                    if list_idx < arrayLength(&light_index_list) {
                        light_index_list[list_idx] = i;
                    }
                    cluster_light_count += 1u;
                }
            }
        }

        // Test spot lights
        for (var i = 0u; i < params.num_spot_lights; i = i + 1u) {
            let sl = spot_lights[i];
            if cone_intersects_frustum_cluster(
                sl.position, sl.direction, sl.radius, sl.cos_outer,
                vec2<f32>(u_min, v_min), vec2<f32>(u_max, v_max),
                depth_near, depth_far,
            ) {
                if cluster_light_count < MAX_LIGHTS_PER_CLUSTER {
                    let list_idx = light_count + cluster_light_count;
                    if list_idx < arrayLength(&light_index_list) {
                        light_index_list[list_idx] = i + params.num_point_lights;
                    }
                    cluster_light_count += 1u;
                }
            }
        }

        // Write offset and count for this cluster
        if slice_start + 1u < arrayLength(&light_grid) {
            light_grid[slice_start] = light_count;
            light_grid[slice_start + 1u] = cluster_light_count;
        }

        light_count += cluster_light_count;
    }
}

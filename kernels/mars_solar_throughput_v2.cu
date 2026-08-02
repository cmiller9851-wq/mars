#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define TILE_DIM 16
#define SMEM_STRIDE (TILE_DIM + 2) // 18 elements avoids 32-bank alignment stride conflicts

__constant__ float d_SOLAR_CONSTANT = 590.0f; // Mars Solar Constant (W/m^2)

// Host-calculated invariant parameters passed as POD struct
struct SolarParams {
    float sx;           // sin(zenith) * sin(azimuth)
    float sy;           // sin(zenith) * cos(azimuth)
    float sz;           // cos(zenith)
    float direct_flux;  // Solar Constant * exp(-dust_tau * air_mass)
    float inv_cell_size; // 1.0f / cell_resolution_meters
};

__global__ void __launch_bounds__(256, 2) mars_solar_throughput_kernel_v2(
    const float* __restrict__ dem_grid,   // Input elevation grid
    float* __restrict__ irradiance_out,   // Output flux map
    const int width,
    const int height,
    const SolarParams params
) {
    // 1D contiguous shared memory layout mapped with explicit non-conflicting stride
    __shared__ float smem_tile[SMEM_STRIDE * SMEM_STRIDE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_DIM + tx;
    const int gy = blockIdx.y * TILE_DIM + ty;

    const int sm_x = tx + 1;
    const int sm_y = ty + 1;

    // Coalesced global load for main tile body
    const bool in_bounds = (gx < width) && (gy < height);
    const float center_val = in_bounds ? dem_grid[gy * width + gx] : 0.0f;
    smem_tile[sm_y * SMEM_STRIDE + sm_x] = center_val;

    // Parallel boundary halo fetch - avoids serialized bank conflicts
    if (tx == 0) {
        int left_x = max(gx - 1, 0);
        smem_tile[sm_y * SMEM_STRIDE + 0] = (gy < height) ? dem_grid[gy * width + left_x] : center_val;
    }
    if (tx == TILE_DIM - 1 || gx == width - 1) {
        int right_x = min(gx + 1, width - 1);
        smem_tile[sm_y * SMEM_STRIDE + (TILE_DIM + 1)] = (gy < height) ? dem_grid[gy * width + right_x] : center_val;
    }
    if (ty == 0) {
        int top_y = max(gy - 1, 0);
        smem_tile[0 * SMEM_STRIDE + sm_x] = (gx < width) ? dem_grid[top_y * width + gx] : center_val;
    }
    if (ty == TILE_DIM - 1 || gy == height - 1) {
        int bottom_y = min(gy + 1, height - 1);
        smem_tile[(TILE_DIM + 1) * SMEM_STRIDE + sm_x] = (gx < width) ? dem_grid[bottom_y * width + gx] : center_val;
    }

    __syncthreads();

    if (!in_bounds) return;

    const int idx = gy * width + gx;

    // Horizon check: If sun is below horizon, skip spatial normal calculations
    if (params.sz <= 0.0f) {
        irradiance_out[idx] = 0.0f;
        return;
    }

    // Finite difference grid stencil with scale mapping
    const float z_east  = smem_tile[sm_y * SMEM_STRIDE + (sm_x + 1)];
    const float z_west  = smem_tile[sm_y * SMEM_STRIDE + (sm_x - 1)];
    const float z_south = smem_tile[(sm_y + 1) * SMEM_STRIDE + sm_x];
    const float z_north = smem_tile[(sm_y - 1) * SMEM_STRIDE + sm_x];

    const float dz_dx = (z_east - z_west) * 0.5f * params.inv_cell_size;
    const float dz_dy = (z_south - z_north) * 0.5f * params.inv_cell_size;

    // Compute surface normal with fast inverse square root intrinsic
    const float inv_norm = __rsqrtf(dz_dx * dz_dx + dz_dy * dz_dy + 1.0f);
    const float nx = -dz_dx * inv_norm;
    const float ny = -dz_dy * inv_norm;
    const float nz = inv_norm;

    // Directional Dot Product (N . S)
    const float cos_theta = fmaxf(0.0f, (nx * params.sx) + (ny * params.sy) + (nz * params.sz));

    // Final irradiance mapping
    irradiance_out[idx] = params.direct_flux * cos_theta;
}

// C-ABI Host Wrapper Function for Drop-In Execution
extern "C" {

void launch_mars_solar_throughput(
    const float* d_dem_grid,
    float* d_irradiance_out,
    int width,
    int height,
    SolarParams params,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    mars_solar_throughput_kernel_v2<<<grid, block, 0, stream>>>(
        d_dem_grid,
        d_irradiance_out,
        width,
        height,
        params
    );
}

} // extern "C"

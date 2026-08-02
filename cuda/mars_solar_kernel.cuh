#ifndef MARS_SOLAR_KERNEL_CUH
#define MARS_SOLAR_KERNEL_CUH

#include <cuda_runtime.h>

#define TILE_DIM 16
#define SMEM_STRIDE (TILE_DIM + 2) // 18-element stride prevents shared memory bank conflicts

struct SolarParams {
    float sx;            // sin(zenith) * sin(azimuth)
    float sy;            // sin(zenith) * cos(azimuth)
    float sz;            // cos(zenith)
    float direct_flux;   // W/m^2 adjusted for Beer-Lambert attenuation
    float inv_cell_size; // 1.0f / cell_resolution_meters
};

extern "C" {
    void launch_mars_solar_throughput(
        const float* d_dem_grid,
        float* d_irradiance_out,
        int width,
        int height,
        SolarParams params,
        cudaStream_t stream
    );
}

#endif // MARS_SOLAR_KERNEL_CUH

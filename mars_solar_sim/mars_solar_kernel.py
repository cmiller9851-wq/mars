import numpy as np
import mmap
import os
import time
from .dependencies import verify_environment
from .dem_generator import generate_synthetic_dem

# Mars Physical Constants
SOLAR_CONSTANT_MARS = 590.0  # W/m^2 (Average solar flux at Mars orbital distance)

def execute_solar_irradiance_kernel(
    dem_filepath="mars_terrain.bin",
    grid_size=2048,
    solar_zenith_deg=30.0,
    solar_azimuth_deg=135.0,
    dust_opacity_tau=0.5,
    cell_spacing_meters=1.0
):
    """
    Evaluates direct solar irradiance on terrain facets using vectorized SIMD math.
    Memory-maps the target file to avoid loading entire datasets into local RAM.
    """
    if not os.path.exists(dem_filepath):
        generate_synthetic_dem(dem_filepath, grid_size)

    file_size = os.path.getsize(dem_filepath)
    expected_bytes = grid_size * grid_size * 4  # float32 = 4 bytes
    
    if file_size != expected_bytes:
        raise ValueError(f"File size ({file_size} bytes) does not match grid dimension specifications.")

    print(f"Memory-mapping DEM binary: {dem_filepath} ...")
    start_time = time.time()

    with open(dem_filepath, "r+b") as f:
        # Zero-copy memory mapping
        mm = mmap.mmap(f.fileno(), 0)
        
        # Wrap the byte stream in a NumPy array buffer without copying data to RAM
        dem_data = np.frombuffer(mm, dtype=np.float32).reshape((grid_size, grid_size))

        # 1. Convert Solar Angles to Vector Direction Components
        zenith_rad = np.radians(solar_zenith_deg)
        azimuth_rad = np.radians(solar_azimuth_deg)

        # Sun unit vector s = [sx, sy, sz]
        s_x = np.sin(zenith_rad) * np.sin(azimuth_rad)
        s_y = np.sin(zenith_rad) * np.cos(azimuth_rad)
        s_z = np.cos(zenith_rad)

        # 2. Vectorized Gradient Surface Normal Computation (3x3 finite difference)
        # dz/dx and dz/dy calculated using NumPy's gradient implementation
        dy, dx = np.gradient(dem_data, cell_spacing_meters)

        # Normal vectors N = [-dz/dx, -dz/dy, 1] / magnitude
        norm_factor = np.sqrt(dx**2 + dy**2 + 1.0)
        n_x = -dx / norm_factor
        n_y = -dy / norm_factor
        n_z = 1.0 / norm_factor

        # 3. Surface Dot Product (cos theta_i = N dot S)
        dot_product = (n_x * s_x) + (n_y * s_y) + (n_z * s_z)
        
        # Clamp negative values (self-shadowed facets facing away from the sun)
        cos_inc_angle = np.clip(dot_product, 0.0, None)

        # 4. Atmospheric Extinction (Beer-Lambert atmospheric attenuation)
        air_mass = 1.0 / np.cos(zenith_rad)
        direct_flux = SOLAR_CONSTANT_MARS * np.exp(-dust_opacity_tau * air_mass)

        # 5. Total Irradiance Calculation per Facet
        irradiance_map = direct_flux * cos_inc_angle

        # Flush compute pass
        mm.close()

    elapsed = time.time() - start_time
    throughput_mcells = (grid_size * grid_size) / (elapsed * 1e6)

    print(f"=== KERNEL EXECUTION COMPLETE ===")
    print(f"Processed: {grid_size}x{grid_size} facets ({grid_size**2:,} points)")
    print(f"Compute Time: {elapsed:.4f} seconds")
    print(f"Throughput: {throughput_mcells:.2f} Million Facets/sec")
    print(f"Max Local Flux: {np.max(irradiance_map):.2f} W/m^2")
    print(f"Mean Local Flux: {np.mean(irradiance_map):.2f} W/m^2")

    return irradiance_map

if __name__ == "__main__":
    verify_environment()
    execute_solar_irradiance_kernel()
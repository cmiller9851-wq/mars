import numpy as np
import os

def generate_synthetic_dem(filename="mars_terrain.bin", grid_size=2048):
    """
    Creates a flat binary file of float32 terrain elevations.
    A 2048x2048 grid produces a 16MB file. Scaling grid_size increases tile size linearly.
    """
    print(f"Generating synthetic binary DEM: {filename} ({grid_size}x{grid_size})...")
    
    # Create synthetic crater/slope terrain topography
    x = np.linspace(-10, 10, grid_size, dtype=np.float32)
    y = np.linspace(-10, 10, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Mathematical elevation surface: Z = synthetic hills and crater depressions
    Z = (np.sin(X) * np.cos(Y) * 150.0) - (np.exp(-(X**2 + Y**2)/10.0) * 500.0)
    Z = Z.astype(np.float32)

    # Write directly as raw byte buffer to file
    with open(filename, "wb") as f:
        f.write(Z.tobytes())
        
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"DEM binary successfully written: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    generate_synthetic_dem()
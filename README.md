# mars

High-throughput Mars terrain + solar irradiance pipeline.

Includes:
- mars_solar_sim/ : Python components for DEM generation, memory-mapped irradiance compute, Starlink telemetry probe, and CCSDS packetizer
- cuda/ : Optimized CUDA kernel and host launcher for high-throughput execution
- LICENSE-SAEL.txt : Project license (SOVEREIGN AUTHORSHIP ENFORCED LICENSE v1.0)

Quick start (local):
1. Clone:
   git clone https://github.com/cmiller9851-wq/mars.git
   cd mars

2. Python: create a virtualenv and install numpy
   python -m venv venv
   source venv/bin/activate
   pip install numpy

3. Run the memory-mapped CPU kernel (example):
   python -m mars_solar_sim.mars_solar_kernel

4. CUDA:
   - Build the CUDA files with nvcc or CMake if you have CUDA installed.
   - Example (nvcc):
     nvcc -arch=sm_70 cuda/mars_solar_kernel.cu -o cuda/mars_solar_kernel.o
     g++ -shared -o cuda/libmars_solar.so cuda/mars_solar_kernel.o cuda/launch_kernel.cpp -lcudart

Notes and warnings:
- starlink_telemetry.py performs network probes to a Starlink transceiver IP (default 192.168.100.1:9200). Ensure you have authorization to connect to the device and comply with SpaceX/Starlink terms of service.
- The SAEL license restricts copying, redistribution, training AI systems, or creating derivative commercial works without prior written authorization. Confirm license terms and applicability for contributors and consumers.
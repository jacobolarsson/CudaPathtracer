# CUDA Pathtracer
## Overview
GPU-accelerated rendering engine built from scratch using C++ and CUDA. It implements a pathtracing algorithm to simulate realistic lighting and global illumination by tracing the paths of light rays in a 3D scene. The acceleration provided by using the GPU through CUDA combined with the use of space partitioning data structures provide a more than x100 speedup in execution time compared to a single-threaded version of the program.

## Key features 
- Physically-Based Rendering: The engine calculates light interactions with surfaces using principles of physics, producing visually accurate reflections, refractions, and shadows.
- High Performance: By leveraging CUDA, the rendering process is massively parallelized by the GPU, and the use of a kd-tree space partitioning data structure highly optimizes ray intersections.
- Custom Materials: Supports various material types, including metallic, diffusive, and refractive.

## Results

![SphereGlass_s100](https://github.com/user-attachments/assets/d689cea4-51ac-4428-9ca3-68d7564fe671)

128 rays per pixel and 10 bounces per ray, 1024x1024 resolution:
- Single threaded: 9 minutes 35 seconds
- Using CUDA: 5.259 seconds

![SuzanneGlass](https://github.com/user-attachments/assets/0e9e1c07-9717-448a-81bf-145984c21002)

128 rays per pixel and 10 bounces per ray, 1024x1024 resolution:
- Single threaded: 12 minutes 41 seconds
- Using CUDA: 7.142 seconds

![BunnyGlass](https://github.com/user-attachments/assets/c0a87a89-cfa3-407d-a33b-9d1a3dcc7b38)

256 rays per pixel and 8 bounces per ray, 1024x1024 resolution:
- Single threaded: 21 minutes 56 seconds
- Using CUDA: 12.577 seconds

## Future work
- Improve bandwidth usage by performing parallel memory transfers.
- Test different types of VRAM memory allocation.
- Make the CUDA render kernels run in parallel.
- Use CUDA shared memory to avoid cache misses.
- Implement other raytracing features.
- 


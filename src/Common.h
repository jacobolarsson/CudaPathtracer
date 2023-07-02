#ifndef COMMON_H
#define COMMON_H

#include "../include/SDL2/SDL.h"
#include "../include/imgui/imgui_impl_sdl2.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

namespace Raytracer
{
    #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
    void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
        if (result) {
            std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                file << ":" << line << " '" << func << "' \n";
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            exit(99);
        }
    }
}

#endif
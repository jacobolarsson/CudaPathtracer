#pragma once
#ifndef COMMON_H
#define COMMON_H

#define GLM_FORCE_CUDA

#include "../include/SDL2/SDL.h"
#include "../include/imgui/imgui_impl_sdl2.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "../include/glm/gtc/matrix_transform.hpp"
#include "../include/glm/gtx/norm.hpp"
#include "../include/glm/gtc/random.hpp"
#include "../include/glm/gtx/euler_angles.hpp"
#include <iostream>

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat2;
using glm::mat3;
using glm::mat4;

namespace Raytracer
{
	#define checkCuda(val) check_cuda( (val), #val, __FILE__, __LINE__ )
	#define checkCuRandErrors(val) check_curand( (val), #val, __FILE__, __LINE__ )

	void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);
	void check_curand(cudaError_t result, char const* const func, const char* const file, int const line);
}
#endif
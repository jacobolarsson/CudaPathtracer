#include "Common.h"

void Raytracer::check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA error:" << result << ": " << cudaGetErrorString(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Free all device memory
		cudaDeviceReset();
		exit(99);
	}
}

void Raytracer::check_curand(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result != CURAND_STATUS_SUCCESS) {
		std::cerr << "cuRAND error" << result << ": " << cudaGetErrorString(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Free all device memory
		cudaDeviceReset();
		exit(99);
	}
}
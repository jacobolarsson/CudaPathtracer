/*!
 * @file main.cpp
 *
 * @author jacobo.larsson
 * @date 07/22/2023
 * @brief This project contains a raytracer renderer
 *
 */

#define SDL_MAIN_HANDLED

#include "Renderer/Renderer.h"
#include "Parser/Parser.h"

#include <chrono>

using namespace Raytracer;
using Timer = std::chrono::high_resolution_clock;

void LogTime(const char* what, Timer::time_point start, Timer::time_point end)
{
	std::cout 
		<< what 
		<< " took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
		<< " ms"
		<< std::endl;
}

int main(int argc, char* argv[])
{
    const int textureWidth = atoi(argv[1]);
    const int textureHeight = atoi(argv[2]);

    const int windowWidth = 1024;
    const int windowHeight = 1024;

	Scene* dScene;
	checkCuda(cudaMalloc(&dScene, sizeof(Scene)));
	
	Parser parser("A4_Box_Transmission.txt", dScene);
	GameObject** dObjects;
	KdTree kdTree;
	int size = parser.GetObjectCount();

	// Allocate memory for the scene and the kd-tree
	checkCuda(cudaMalloc(&dObjects, sizeof(GameObject*) * size));

	DeviceFunc::CreateScene<<<1, 1>>>(dScene, dObjects, size);
	checkCuda(cudaDeviceSynchronize());

	Timer::time_point start;
	// Populate the scene using data from a given scene file
	parser.LoadScene(&kdTree);
	start = Timer::now();
	kdTree.CreateTree();
	LogTime("Creating the kdTree", start, Timer::now());

	int nodeCount = static_cast<int>(kdTree.GetNodes().size());
	int primCount = static_cast<int>(kdTree.GetPrimitives().size());
	int meshCount = static_cast<int>(kdTree.GetMeshes().size());
	// Allocate device memory for the kd-tree nodes and triagles

	KdNode* dKdNodes;
	Triangle* dPrimitives;
	BoundingVolume* dBoundingVol;
	Mesh* dMeshes;
	start = Timer::now();
	checkCuda(cudaMalloc(&dKdNodes, nodeCount * sizeof(KdNode)));
	checkCuda(cudaMalloc(&dPrimitives, primCount * sizeof(Triangle)));
	checkCuda(cudaMalloc(&dBoundingVol, sizeof(BoundingVolume)));
	checkCuda(cudaMalloc(&dMeshes, sizeof(Mesh) * kdTree.GetMeshes().size()));

	// Send the kd-tree data to the device allocated scene
	checkCuda(cudaMemcpy(dKdNodes, kdTree.GetNodes().data(), nodeCount * sizeof(KdNode), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dPrimitives, kdTree.GetPrimitives().data(), primCount * sizeof(Triangle), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dBoundingVol, &kdTree.GetBoundingVolume(), sizeof(BoundingVolume), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dMeshes, kdTree.GetMeshes().data(), sizeof(Mesh) * meshCount, cudaMemcpyHostToDevice));

	DeviceFunc::SetKdNodes<<<1, 1>>>(dScene, dKdNodes, nodeCount, dPrimitives, primCount, dBoundingVol, dMeshes);
	checkCuda(cudaDeviceSynchronize());

	LogTime("Copying the kdTree into DRAM", start, Timer::now());
	std::cout << std::endl;

	Renderer renderer(windowWidth,
					  windowHeight,
					  textureWidth,
					  textureHeight,
					  dScene, parser.GetCamera());

	Timer::time_point totalStart = Timer::now();
	SDL_Event e;
	bool quit = false;

	for (int i = 0; i < Renderer::cSquareDivisionsSqrt; i++) {
		for (int j = 0; j < Renderer::cSquareDivisionsSqrt; j++) {
			start = Timer::now();
			renderer.ComputeSceneTexture(j, i);
			renderer.PresentCurrentSceneTexture(j, i);

			LogTime(std::string("Rendering square[" +
								std::to_string(i) +
								"][" + std::to_string(j) +
								"]").c_str(), start, Timer::now());
		}
	}
	LogTime("\nTotal rendering", totalStart, Timer::now());

	while (!quit) {
		SDL_WaitEvent(&e);
		if (e.type == SDL_QUIT) {
			quit = true;
		}
		renderer.PresentSceneTexture();
	}
	renderer.ExportTexture();

	DeviceFunc::ClearScene<<<1, 1>>>(dScene);
	checkCuda(cudaDeviceSynchronize());
	checkCuda(cudaFree(dScene));
	checkCuda(cudaFree(dObjects));

    return 0;
}
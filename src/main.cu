/*!
 * @file main.cpp
 *
 * @author jacobo.larsson
 * @date 05/22/2023
 * @brief This project contains a raytracer renderer
 *
 */

#define SDL_MAIN_HANDLED

#include "Renderer/Renderer.h"
#include "Parser/Parser.h"

using namespace Raytracer;

int main(int argc, char* argv[])
{
    const int textureWidth = atoi(argv[1]);
    const int textureHeight = atoi(argv[2]);

    const int windowWidth = 1024;
    const int windowHeight = 1024;

	Scene* dScene;
	checkCudaErrors(cudaMalloc(&dScene, sizeof(Scene)));
	
	Parser parser("A3_Suzanne.txt", dScene);
	GameObject** dObjects;
	int size = parser.GetObjectCount();

	checkCudaErrors(cudaMalloc(&dObjects, sizeof(GameObject*) * size));

	DeviceFunc::CreateScene<<<1, 1>>>(dScene, dObjects, size);

	parser.LoadScene();

	Renderer renderer(windowWidth, windowHeight, textureWidth, textureHeight, dScene, parser.GetCamera());

	clock_t start, stop;
	start = clock();
	SDL_Event e;
	bool quit = false;
	for (int i = 0; i < renderer.GetConfig().samples; i += 10) {
		renderer.ComputeSceneTexture();

		checkCudaErrors(cudaDeviceSynchronize());
		renderer.PresentCurrentSceneTexture(i);

		stop = clock();
		double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
		std::cout << timer_seconds << " seconds.\n";
	}

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	renderer.PresentCurrentSceneTexture(renderer.GetConfig().samples - 1);
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << timer_seconds << " seconds.\n";
	std::cout << "FINISHED\n";

	while (!quit) {
		SDL_WaitEvent(&e);
		if (e.type == SDL_QUIT) {
			quit = true;
		}

		renderer.PresentSceneTexture();
	}

	DeviceFunc::ClearScene<<<1, 1>>>(dScene);
	checkCudaErrors(cudaFree(dScene));
	checkCudaErrors(cudaFree(dObjects));

    return 0;
}
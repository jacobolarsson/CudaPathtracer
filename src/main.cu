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
	renderer.ComputeSceneTexture();
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "took " << timer_seconds << " seconds.\n";

	renderer.ExportTexture();

	SDL_Event e;
	bool quit = false;

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
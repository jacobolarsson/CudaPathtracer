/*!
 * @file main.cpp
 *
 * @author jacobo.larsson
 * @date 05/22/2023
 * @brief This project contains a raytracer renderer
 *
 */

#define SDL_MAIN_HANDLED

#include "Renderer/Renderer.cu"

#include <string>

using namespace Raytracer;

int main(int argc, char* argv[])
{
    const int textureWidth = atoi(argv[1]);
    const int textureHeight = atoi(argv[2]);

    const int windowWidth = 512;
    const int windowHeight = 512;

	Renderer renderer(windowWidth, windowHeight, textureWidth, textureHeight);
	renderer.ComputeSceneTexture();

	SDL_Event e;
	bool quit = false;

	while (!quit) {
		SDL_WaitEvent(&e);
		if (e.type == SDL_QUIT) {
			quit = true;
		}
		renderer.PresentSceneTexture();
	}

    return 0;
}
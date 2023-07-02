#ifndef RENDERER_H
#define RENDERER_H
#include "../Math.h"

namespace Raytracer
{
	struct Pixel
	{
		__host__ __device__ Pixel() : r(0), g(0), b(0), a(255) {}
		__host__ __device__ Pixel(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha)
			: r(red)
			, g(green)
			, b(blue)
			, a(alpha)
		{}

		__host__ __device__ Pixel(vec3 rgbVec)
		{
			r = static_cast<unsigned char>(255.99f * clamp(0.0f, 0.999f, rgbVec.x));
			g = static_cast<unsigned char>(255.99f * clamp(0.0f, 0.999f, rgbVec.y));
			b = static_cast<unsigned char>(255.99f * clamp(0.0f, 0.999f, rgbVec.z));
			a = 255;
		}

		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;
	};

	class Renderer
	{
	public:
		Renderer(unsigned windowWidth,
			unsigned windowHeight,
			unsigned textureWidth,
			unsigned textureHeight,
			int threadXCount = 8,
		    int threadYCount = 8);

		~Renderer();

		void ComputeSceneTexture() const;
		void PresentSceneTexture();
		void ExportTexture() const;

	private:
		unsigned mWindowWidth;
		unsigned mWindowHeigth;
		unsigned mTextureWidth;
		unsigned mTextureHeight;
		SDL_Window* mWindow;
		SDL_Renderer* mRenderer;
		SDL_Texture* mTexture;
		Pixel* mFrameBuffer;
		const int mThreadXCount;
		const int mThreadYCount;
	};

}

#endif
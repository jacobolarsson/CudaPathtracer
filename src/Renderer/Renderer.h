#pragma once

#include "../Math.h"
#include "../Scene/Scene.h"
#include "../Camera/Camera.h"

namespace Raytracer
{
	struct RenderConfig
	{
		int samples = 1;
		int bounces = 1;
	};

	struct Pixel
	{
		__host__ __device__ Pixel() : r(0), g(0), b(0), a(255) {}
		__host__ __device__ Pixel(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha)
			: r(red)
			, g(green)
			, b(blue)
			, a(alpha)
		{}

		__host__ __device__ Pixel(vec3 rgbVec, int samples = 1)
		{
			float scale = 1.0f / samples;
			rgbVec.x *= scale;
			rgbVec.y *= scale;
			rgbVec.z *= scale;

			r = static_cast<unsigned char>(255.99f * glm::clamp(rgbVec.x, 0.0f, 0.999f));
			g = static_cast<unsigned char>(255.99f * glm::clamp(rgbVec.y, 0.0f, 0.999f));
			b = static_cast<unsigned char>(255.99f * glm::clamp(rgbVec.z, 0.0f, 0.999f));
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
			Scene* dScene,
			std::unique_ptr<Camera>& camera,
			int threadXCount = 8,
		    int threadYCount = 8);

		~Renderer();

		void Init();
		void ComputeSceneTexture()const;
		void PresentCurrentSceneTexture(int currentSamples);
		void PresentSceneTexture();
		void ExportTexture() const;

		RenderConfig const& GetConfig() const { return m_config; }

	private:
		void ReadRenderConfig();

		unsigned m_windowWidth;
		unsigned m_windowHeight;
		unsigned m_textureWidth;
		unsigned m_textureHeight;
		SDL_Window* m_window;
		SDL_Renderer* m_renderer;
		SDL_Texture* m_texture;
		vec3* m_frameBufferVec;
		Pixel* m_frameBuffer;
		Scene* m_dScene;
		curandState* m_dRandStates;
		std::unique_ptr<Camera> m_camera;
		const int m_threadXCount;
		const int m_threadYCount;
		RenderConfig m_config;
	};
	namespace DeviceFunc
	{
		__global__ void render(vec3* fb,
			Scene* dScene,
			int max_x,
			int max_y,
			vec3 lower_left_corner,
			vec3 horizontal,
			vec3 vertical,
			vec3 origin,
			curandState* dRandStates,
			int bounces);

		__global__ void InitRandStates(int textureWidth, int textureHeight, curandState* dRandStates);
	}
}
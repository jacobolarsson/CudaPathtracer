#include "Renderer.h"

#include <fstream>
#include <time.h>

using namespace Raytracer;

Renderer::Renderer(unsigned windowWidth,
				   unsigned windowHeight,
				   unsigned textureWidth,
				   unsigned textureHeight,
				   Scene* dScene,
	    		   std::unique_ptr<Camera>& camera,
				   int threadXCount,
				   int threadYCount)
	: m_windowWidth(windowWidth)
	, m_windowHeight(windowHeight)
	, m_textureWidth(textureWidth)
	, m_textureHeight(textureHeight)
	, m_window(nullptr)
	, m_renderer(nullptr)
	, m_texture(nullptr)
	, m_frameBufferVec(nullptr)
	, m_frameBuffer(nullptr)
	, m_dScene(dScene)
	, m_dRandStates(nullptr)
	, m_camera(std::move(camera))
	, m_threadXCount(threadXCount)
	, m_threadYCount(threadYCount)
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		throw std::exception("Could not initialize SDL");
	}

	// Create the window, the renderer and the texture to render
	m_window = SDL_CreateWindow("Cuda Raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight, SDL_WINDOW_SHOWN);
	m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED);
	m_texture = SDL_CreateTexture(m_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, textureWidth, textureHeight);

	if (!m_window || !m_renderer || !m_texture) {
		throw std::exception("Could not create SDL rendering framework");
	}
	// Allocate memory for the frame buffer
	size_t pixelCount = static_cast<size_t>(m_textureWidth) * static_cast<size_t>(m_textureHeight);
	m_frameBuffer = new Pixel[pixelCount];

	checkCudaErrors(cudaHostAlloc(&m_frameBufferVec, pixelCount * sizeof(vec3), cudaHostAllocMapped));
	checkCudaErrors(cudaMalloc(&m_dRandStates, pixelCount * sizeof(curandState)));

	for (int i = 0; i < pixelCount; i++) {
		m_frameBufferVec[i] = { 0.0f, 0.0f, 0.0f };
	}

	ReadRenderConfig();
	Init();
}

Renderer::~Renderer()
{
	checkCudaErrors(cudaFreeHost(m_frameBufferVec));
	SDL_DestroyWindow(m_window);
	SDL_DestroyRenderer(m_renderer);
	SDL_DestroyTexture(m_texture);
	SDL_Quit();
}

void Renderer::Init()
{
	dim3 blocks(m_textureWidth / m_threadXCount, m_textureHeight / m_threadYCount);
	dim3 threads(m_threadXCount, m_threadYCount);

	DeviceFunc::InitRandStates<<<blocks, threads>>>(m_textureWidth, m_textureHeight, m_dRandStates);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void Renderer::ComputeSceneTexture() const
{
	dim3 blocks(m_textureWidth / m_threadXCount, m_textureHeight / m_threadYCount);
	dim3 threads(m_threadXCount, m_threadYCount);

	vec3 targetVec = normalize(m_camera->target - m_camera->pos);
	vec3 horizontal = normalize(cross(targetVec, m_camera->up));
	vec3 vertical = normalize(cross(horizontal, targetVec));
	vec3 lowerLeftCorner = m_camera->pos - horizontal / 2.0f - vertical / 2.0f + targetVec * m_camera->focal;

	vec3* deviceFb;
	checkCudaErrors(cudaHostGetDevicePointer(&deviceFb, m_frameBufferVec, 0));

	DeviceFunc::render<<<blocks, threads>>>(deviceFb,
										    m_dScene,
										    m_textureWidth,
										    m_windowHeight,
										    lowerLeftCorner,
										    horizontal,
										    vertical,
										    m_camera->pos,
										    m_dRandStates,
										    m_config.bounces);
}

void Renderer::PresentCurrentSceneTexture(int currentSamples)
{
	size_t pixelCount = static_cast<size_t>(m_textureWidth) * static_cast<size_t>(m_textureHeight);
	for (int i = 0; i < pixelCount; i++) {
		m_frameBuffer[i] = Pixel(m_frameBufferVec[i], currentSamples);
	}

	SDL_UpdateTexture(m_texture, nullptr, m_frameBuffer, sizeof(Pixel) * m_textureWidth);
	SDL_RenderClear(m_renderer);
	SDL_RenderCopy(m_renderer, m_texture, nullptr, nullptr);
	// Render
	SDL_RenderPresent(m_renderer);
}

void Renderer::PresentSceneTexture()
{
	SDL_RenderClear(m_renderer);
	SDL_RenderCopy(m_renderer, m_texture, nullptr, nullptr);
	// Render
	SDL_RenderPresent(m_renderer);
}

void Renderer::ExportTexture() const
{
	SDL_Texture* target = SDL_GetRenderTarget(m_renderer);
	SDL_SetRenderTarget(m_renderer, m_texture);
	SDL_Surface* surface = SDL_CreateRGBSurface(0, m_windowWidth, m_windowHeight, 32, 0, 0, 0, 0);
	SDL_RenderReadPixels(m_renderer, nullptr, surface->format->format, surface->pixels, surface->pitch);
	SDL_SaveBMP(surface, "rendered_image.bmp");

	SDL_FreeSurface(surface);
	SDL_SetRenderTarget(m_renderer, target);
}

void Renderer::ReadRenderConfig()
{
	std::ifstream file("config.txt");
	if (!file.is_open()) {
		throw std::runtime_error("Could not open config.txt file");
	}

	std::string str{};
	while (!file.eof()) {
		str = "";
		file >> str;

		if (str == "SAMPLES") {
			str = "";
			file >> str;
			m_config.samples = std::atoi(str.c_str());
		}
		else if (str == "BOUNCES") {
			str = "";
			file >> str;
			m_config.bounces = std::atoi(str.c_str());
		}
	}
}

__global__ void DeviceFunc::render(vec3* fb,
								  Scene* dScene,
								  int textureWidth,
								  int textureHeight,
								  vec3 lowerLeftCorner,
								  vec3 horizontal,
								  vec3 vertical,
								  vec3 origin,
								  curandState* dRandStates,
								  int bounces)
{
	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	if ((xIdx >= textureWidth) || (yIdx >= textureHeight)) {
		return;
	}
	int pixelIdx = (textureHeight - yIdx - 1) * textureWidth + xIdx;

	vec3 color = { 0.0f, 0.0f, 0.0f };
	for (int i = 0; i < 10; i++)
	{
		float u = (static_cast<float>(xIdx) + curand_uniform(&dRandStates[pixelIdx])) / textureWidth;
		float v = (static_cast<float>(yIdx) + curand_uniform(&dRandStates[pixelIdx])) / textureHeight;
		Ray r(origin, glm::normalize(lowerLeftCorner + u * horizontal + v * vertical - origin));
		color += dScene->QueryRay(r, bounces, &dRandStates[pixelIdx]);

	}
	fb[pixelIdx] += color;
}

__global__ void DeviceFunc::InitRandStates(int textureWidth, int textureHeight, curandState* dRandStates)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= textureWidth) || (j >= textureHeight)) {
		return;
	}
	int pixelIdx = j * textureWidth + i;
	curand_init(pixelIdx, 0, 0, &dRandStates[pixelIdx]);
}

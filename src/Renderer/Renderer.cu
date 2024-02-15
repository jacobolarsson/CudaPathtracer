#include "Renderer.h"

#include <fstream>
#include <time.h>

using namespace Raytracer;

Renderer::Renderer(unsigned windowWidth,
				   unsigned windowHeight,
				   unsigned textureWidth,
				   unsigned textureHeight,
				   Scene* dScene,
	    		   std::unique_ptr<Camera>& camera)
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
	, m_deviceFb(nullptr)
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

	const int squarePixelCount = m_textureHeight * m_textureWidth / cSquareDivisions;

	checkCuda(cudaHostAlloc(&m_frameBufferVec, pixelCount * sizeof(vec3), cudaHostAllocMapped));
	checkCuda(cudaMalloc(&m_dRandStates, squarePixelCount * cThreadCount * sizeof(curandState)));

	for (int i = 0; i < pixelCount; i++) {
		m_frameBufferVec[i] = { 0.0f, 0.0f, 0.0f };
	}

	checkCuda(cudaHostGetDevicePointer(&m_deviceFb, m_frameBufferVec, 0));

	ReadRenderConfig();
	Init();
}

Renderer::~Renderer()
{
	checkCuda(cudaFreeHost(m_frameBufferVec));
	checkCuda(cudaFree(m_dRandStates));
	SDL_DestroyWindow(m_window);
	SDL_DestroyRenderer(m_renderer);
	SDL_DestroyTexture(m_texture);
	SDL_Quit();
}

void Renderer::Init()
{
	dim3 blocks(m_textureWidth / cSquareDivisionsSqrt, m_textureHeight / cSquareDivisionsSqrt);
	dim3 threads(cThreadCount);

	DeviceFunc::InitRandStates<<<blocks, threads>>>(m_dRandStates);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
}

void Renderer::ComputeSceneTexture(int squareIdxX, int squareIdxY)
{
	static dim3 blocks(m_textureWidth / cSquareDivisionsSqrt, m_textureHeight / cSquareDivisionsSqrt);
	static dim3 threads(cThreadCount);
	static vec3 targetVec = normalize(m_camera->target - m_camera->pos);
	static vec3 horizontal = normalize(cross(targetVec, m_camera->up));
	static vec3 vertical = normalize(cross(horizontal, targetVec));
	static vec3 lowerLeftCorner = m_camera->pos - horizontal / 2.0f - vertical / 2.0f + targetVec * m_camera->focal;

	DeviceFunc::render<<<blocks, threads>>>(m_deviceFb,
											m_dScene,
											m_textureWidth,
											m_windowHeight,
											lowerLeftCorner,
											horizontal,
											vertical,
											m_camera->pos,
											m_dRandStates,
											m_config,
											squareIdxX,
											squareIdxY);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
}

void Renderer::PresentCurrentSceneTexture(int squareIdxX, int squareIdxY)
{
	int squareWidth = m_textureWidth / cSquareDivisionsSqrt;
	int squareHeight = m_textureHeight / cSquareDivisionsSqrt;
	// Inverse the square index so that we start from the bottom
	squareIdxY = cSquareDivisionsSqrt - 1 - squareIdxY;
	// Copy the content of the given square of pixels into the frame buffer
	for (int i = 0; i < squareHeight; i++) {
		for (int j = 0; j < squareWidth; j++) {
			int squareGridOff = squareIdxY * m_textureWidth * squareHeight + squareIdxX * squareWidth;
			int idx = squareGridOff + i * m_textureWidth + j;
			m_frameBuffer[idx] = Pixel(m_frameBufferVec[idx], m_config.samples);
		}
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
			// Sample count must be a multiple of the thread count per block
			m_config.samples = ((std::atoi(str.c_str()) + cThreadCount - 1) / cThreadCount) * cThreadCount;
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
								  RenderConfig config,
								  int squareIdxX,
								  int squareIdxY)
{
	const int iterations = config.samples / Renderer::cThreadCount;
	const int squareWidth = textureWidth / Renderer::cSquareDivisionsSqrt;
	// Screen indices
	int xIdx = squareIdxX * squareWidth + blockIdx.x;
	int yIdx = squareIdxY * squareWidth + blockIdx.y;

	if ((xIdx >= textureWidth) || (yIdx >= textureHeight)) {
		return;
	}

	int randStateIdx = blockIdx.y * gridDim.x + blockIdx.x + threadIdx.x;

	int pixelIdx = (textureHeight - yIdx - 1) * textureWidth + xIdx;
	// Shared memory variable to accumulate the color value
	__shared__ vec3 color[Renderer::cThreadCount];
	color[threadIdx.x] = { 0.0f, 0.0f, 0.0f };

	vec3 c = { 0.0f, 0.0f, 0.0f };
	// Cast rays
	for (int i = 0; i < iterations; i++) {
		// Apply small random offset to achieve anti-aliasing
		float u = (static_cast<float>(xIdx) + curand_uniform(&dRandStates[randStateIdx])) / textureWidth;
		float v = (static_cast<float>(yIdx) + curand_uniform(&dRandStates[randStateIdx])) / textureHeight;
		Ray r(origin, glm::normalize(lowerLeftCorner + u * horizontal + v * vertical - origin));

		c += dScene->QueryRay(r, config.bounces, &dRandStates[randStateIdx]);
	}
	__syncthreads();
	color[threadIdx.x] = c;
	if (threadIdx.x == 0) {
		vec3 res = { 0.0f, 0.0f, 0.0f };
		// Accumulate the color computed by every thread in this block
		for (int i = 0; i < Renderer::cThreadCount; i++) {
			res += color[i];
		}
		fb[pixelIdx] = res;
	}
}

__global__ void DeviceFunc::InitRandStates(curandState* dRandStates)
{
	int pixelIdx = blockIdx.y * gridDim.x + blockIdx.x + threadIdx.x;
	curand_init(pixelIdx, 0, 0, &dRandStates[pixelIdx]);
}

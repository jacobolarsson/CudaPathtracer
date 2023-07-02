#include "Renderer.cuh"

using namespace Raytracer;

Renderer::Renderer(unsigned windowWidth,
									 unsigned windowHeight,
									 unsigned textureWidth,
									 unsigned textureHeight,
									 int threadXCount,
									 int threadYCount)
	: mWindowWidth(windowWidth)
	, mWindowHeigth(windowHeight)
	, mTextureWidth(textureWidth)
	, mTextureHeight(textureHeight)
	, mThreadXCount(threadXCount)
	, mThreadYCount(threadYCount)
	
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		throw std::exception("Could not initialize SDL");
	}

	// Create the window, the renderer and the texture to render
	mWindow = SDL_CreateWindow("Cuda Raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight, SDL_WINDOW_SHOWN);
	mRenderer = SDL_CreateRenderer(mWindow, -1, SDL_RENDERER_ACCELERATED);
	mTexture = SDL_CreateTexture(mRenderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, textureWidth, textureHeight);

	if (!mWindow || !mRenderer || !mTexture) {
		throw std::exception("Could not create SDL rendering framework");
	}
	// Allocate memory for the frame buffer
	size_t fb_size = static_cast<size_t>(mTextureWidth) * static_cast<size_t>(mTextureHeight) * sizeof(Pixel);
	checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&mFrameBuffer), fb_size));
}

Renderer::~Renderer()
{
	SDL_DestroyWindow(mWindow);
	SDL_DestroyRenderer(mRenderer);
	SDL_DestroyTexture(mTexture);
	SDL_Quit();
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
	vec3 oc = r.orig - center;
	float a = dot(r.dir, r.dir);
	float b = 2.0f * dot(oc, r.dir);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
	if (hit_sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, r)) {
		return vec3(1.0f, 0.0f, 0.0f);
	}
	float t = 0.5f * (r.dir.y + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(Pixel* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;

	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, normalize(lower_left_corner + u * horizontal + v * vertical));
	fb[pixel_index] = Pixel(color(r));
}

void Renderer::ComputeSceneTexture() const
{
	dim3 blocks(mTextureWidth / mThreadXCount, mTextureHeight / mThreadYCount);
	dim3 threads(mThreadXCount, mThreadYCount);

	render<<<blocks, threads>>>(mFrameBuffer, mTextureWidth, mWindowHeigth,
								vec3(-1.0, -1.0, -1.0),
								vec3(2.0, 0.0, 0.0),
								vec3(0.0, 2.0, 0.0),
								vec3(0.0, 0.0, 0.0));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Compute scene texture and set it as the renderer target
	SDL_UpdateTexture(mTexture, nullptr, mFrameBuffer, sizeof(Pixel) * mTextureWidth);
	SDL_RenderClear(mRenderer);
	SDL_RenderCopy(mRenderer, mTexture, nullptr, nullptr);
}

void Renderer::PresentSceneTexture()
{
	SDL_RenderClear(mRenderer);
	SDL_RenderCopy(mRenderer, mTexture, nullptr, nullptr);
	// Render
	SDL_RenderPresent(mRenderer);
}

void Renderer::ExportTexture() const
{
}
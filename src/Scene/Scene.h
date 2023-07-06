#pragma once
#include "../GameObject/GameObject.h"

namespace Raytracer
{
	enum class MaterialType
	{
		LAMBERTIAN,
		METAL,
		LIGHT
	};

	class Scene
	{
	public:


		__device__ void Init(GameObject** objects, int size);
		__device__ void CreateSphere(vec3 pos,
									 float radius,
									 int index,
									 MaterialType matType,
									 vec3 color,
									 float roughness);
		__device__ void CreateBox(vec3 pos,
								  vec3 length,
								  vec3 width,
								  vec3 height,
								  int index,
								  MaterialType matType,
								  vec3 color,
								  float roughness);
		__device__ void CreateMesh(vec3 pos,
								  vec3* vertices,
								  int vtxCount,
								  Face* faces,
								  int facesCount,
								  int index,
								  MaterialType matType,
								  vec3 color,
								  float roughness);
		__device__ vec3 QueryRay(Ray const& ray, int depth, curandState* randState) const;
		__device__ void Clear();

		__device__ GameObject** GetObjects() const { return m_objects; }
		__device__ void SetAmbient(vec3 ambient) { m_ambient = ambient; }

	private:
		__device__ Material* CreateMaterial(MaterialType type, vec3 color, float roughtness);

		GameObject** m_objects = nullptr;
		int m_size = 0;
		vec3 m_ambient = { 0.0f, 0.0f, 0.0f };
	};

	namespace DeviceFunc
	{
		__global__ void CreateScene(Scene* scene, GameObject** objects, int size);
		__global__ void CreateSphere(Scene* scene,
									 vec3 pos,
									 float radius,
									 int index,
									 MaterialType matType,
									 vec3 color,
									 float roughness = 0.0f);

		__global__ void CreateBox(Scene* scene,
								  vec3 pos,
								  vec3 length,
								  vec3 width,
								  vec3 height,
								  int index,
								  MaterialType matType,
								  vec3 color,
								  float roughness = 0.0f);

		__global__ void CreateMesh(Scene* scene,
								   vec3 pos,
								   vec3* vertices,
								   int vtxCount,
								   Face* faces,
								   int faceCount,
								   int index,
								   MaterialType matType,
								   vec3 color,
								   float roughness = 0.0f);

		__global__ void SetAmbient(Scene* scene, vec3 color);
		__global__ void ClearScene(Scene* scene);
	}
}

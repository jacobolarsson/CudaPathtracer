#pragma once
#include "../GameObject/GameObject.h"
#include "../KdTree/KdTree.h"

namespace Raytracer
{
	enum class MaterialType
	{
		LAMBERTIAN,
		METAL,
		DIELECTRIC,
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
									 float roughness,
									 float ior,
									 vec3 att);

		__device__ void CreateBox(vec3 pos,
								  vec3 length,
								  vec3 width,
								  vec3 height,
								  int index,
								  MaterialType matType,
								  vec3 color,
								  float roughness,
								  float ior,
								  vec3 att);

		__device__ void CreatePoly(vec3 pos,
								   vec3* vertices,
								   int vtxCount,
								   int index,
								   MaterialType matType,
								   vec3 color,
								   float roughness,
								   float ior,
								   vec3 att);

		__device__ vec3 QueryRay(Ray const& ray, int depth, curandState* randState) const;
		__device__ void Clear();

		__device__ GameObject** GetObjects() const { return m_objects; }
		__device__ void SetAmbient(vec3 ambient) { m_ambient = ambient; }
		__device__ void SetKdNodes(KdNode* nodes, int kdNodeCount)
		{
			m_kdNodes = nodes;
			m_kdNodeCount = kdNodeCount;
		}
		__device__ void SetPrimitives(Triangle* primitives, int primCount)
		{
			m_primitives = primitives;
			m_primCount = primCount;
		}

		__device__ void SetKdBoundingVol(BoundingVolume* bv) { m_kdBv = *bv; }
		__device__ void SetMeshes(Mesh* meshes) { m_meshes = meshes; }

	private:

		GameObject** m_objects = nullptr;
		int m_size = 0;
		vec3 m_ambient = { 0.0f, 0.0f, 0.0f };
		KdNode* m_kdNodes = nullptr;
		int m_kdNodeCount = 0;
		Triangle* m_primitives = nullptr;
		int m_primCount = 0;
		BoundingVolume m_kdBv{};
		Mesh* m_meshes = nullptr;
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
									 float roughness = 0.0f,
									 float ior = 0.0f,
									 vec3 att = {0.0f, 0.0f, 0.0f});

		__global__ void CreateBox(Scene* scene,
								  vec3 pos,
								  vec3 length,
								  vec3 width,
								  vec3 height,
								  int index,
								  MaterialType matType,
								  vec3 color,
								  float roughness = 0.0f,
							      float ior = 0.0f,
							      vec3 att = {0.0f, 0.0f, 0.0f});

		__global__ void CreatePoly(Scene* scene,
								   vec3 pos,
								   vec3* vertices,
								   int vtxCount,
								   int index,
								   MaterialType matType,
								   vec3 color,
								   float roughness = 0.0f,
								   float ior = 0.0f,
								   vec3 att = {0.0f, 0.0f, 0.0f});

		__global__ void SetAmbient(Scene* scene, vec3 color);
		__global__ void ClearScene(Scene* scene);

		__device__ void CreateMaterial(MaterialType type,
										vec3 color,
										Material*& mat,
										float roughtness = 0.0f,
										float ior = 0.0f,
										vec3 att = {0.0f, 0.0f, 0.0f});

		__global__ void SetKdNodes(Scene* scene,
								   KdNode* kdNodes,
								   int kdNodeCount,
								   Triangle* primitives,
								   int primCount,
								   BoundingVolume* boundingVolume,
								   Mesh* meshes);
	}
}

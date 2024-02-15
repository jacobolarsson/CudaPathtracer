#include "Scene.h"

using namespace Raytracer;

__device__ void Scene::Init(GameObject** objects, int size)
{
	m_objects = objects;
	m_size = size;
}

__device__ void Scene::CreateSphere(vec3 pos,
									float radius,
									int index,
									MaterialType matType,
									vec3 color,
									float roughness,
								    float ior,
								    vec3 att)
{
	Material* mat;
	DeviceFunc::CreateMaterial(matType, color, mat, roughness, ior, att);
	m_objects[index] = new Sphere(pos, mat, radius);
}

__device__ void Scene::CreateBox(vec3 pos,
								 vec3 length,
								 vec3 width,
								 vec3 height,
								 int index,
								 MaterialType matType,
								 vec3 color,
								 float roughness,
								 float ior,
								 vec3 att)
{
	Material* mat;
	DeviceFunc::CreateMaterial(matType, color, mat, roughness, ior, att);
	m_objects[index] = new Box(pos, mat, length, width, height);
}

__device__ void Scene::CreatePoly(vec3 pos,
								  vec3* vertices,
								  int vtxCount,
								  int index,
								  MaterialType matType,
								  vec3 color,
								  float roughness,
								  float ior,
								  vec3 att)
{
	Material* mat;
	DeviceFunc::CreateMaterial(matType, color, mat, roughness, ior, att);
	m_objects[index] = new Polygon(pos, mat, vertices, vtxCount);
}

__device__ vec3 Scene::QueryRay(Ray const& ray, int depth, curandState* randState) const
{
	static const float cInfinity = std::numeric_limits<float>::max();

	vec3 resultColor(1.0f, 1.0f, 1.0f);
	Ray newRay = ray;
	// Check for inersection with all the objects that are not present in the kd-tree
	for (int i = 0; i < depth; i++) {
		HitData closestHit;
		float minT = cInfinity;
		int hitObjIdx = -1;
		// Get the closest hit with an object
		for (int i = 0; i < m_size; i++) {
			HitData hit;
			if (m_objects[i]->RayHits(newRay, hit, minT)) {
				hitObjIdx = i;
				minT = hit.t;
				closestHit = hit;
			}
		}

		HitData hit;
		Mesh* hitMesh = KdTree::RayHits(newRay,
										hit,
										minT,
										m_kdNodes,
										m_kdNodeCount,
										m_primitives,
										m_primCount, 
										m_kdBv,
										m_meshes);
		// No hit
		if (hitObjIdx == -1 && !hitMesh) {
			return resultColor * m_ambient;
		}
		// If an intersection occurs with the kd-tree object, and
		// it's time of instersection is smaller
		if (hitMesh) {
			resultColor = vec3(0.0f, 0.0f, 0.0f);
			continue;
		}
		GameObject* hitObj = m_objects[hitObjIdx];

		// If we are not to bounce the ray, return the object's color multiplied
		// with the accummulated colors
		if (!hitObj->GetMaterial()->BounceRay(newRay, closestHit, newRay, randState)) {
			return resultColor * hitObj->GetMaterial()->GetColor();
		}
		resultColor *= hitObj->GetMaterial()->GetColor();
	}
	return vec3{ 0.0f, 0.0f, 0.0f };
}

__device__ void Raytracer::Scene::Clear()
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	delete[] m_objects;

	cudaFree(m_kdNodes);
	cudaFree(m_primitives);
	cudaFree(&m_kdBv);
	cudaFree(m_meshes);
}

__device__ void DeviceFunc::CreateMaterial(MaterialType type,
										    vec3 color,
										    Material*& mat,
										    float roughtness,
										    float ior,
										    vec3 att)
{
	if (type == MaterialType::LAMBERTIAN) {
		mat = new Lambertian(color);
	} else if (type == MaterialType::METAL) {
		mat = new Metal(color, roughtness);
	} else if (type == MaterialType::DIELECTRIC) {
		mat = new Dielectric(color, ior, att);
	} else if (type == MaterialType::LIGHT) {
		mat = new Light(color);
	}
}

__global__ void DeviceFunc::SetKdNodes(Scene* scene,
									   KdNode* kdNodes,
									   int kdNodeCount,
									   Triangle* primitives,
									   int primCount,
									   BoundingVolume* boundingVolume,
									   Mesh* meshes)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->SetKdNodes(kdNodes, kdNodeCount);
	scene->SetPrimitives(primitives, primCount);
	scene->SetKdBoundingVol(boundingVolume);
	scene->SetMeshes(meshes);
}

__global__ void DeviceFunc::CreateScene(Scene* scene, GameObject** objects, int size)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->Init(objects, size);
}

__global__ void DeviceFunc::CreateSphere(Scene* scene,
										 vec3 pos,
										 float radius,
										 int index,
										 MaterialType matType,
										 vec3 color,
										 float roughness,
										 float ior,
										 vec3 att)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->CreateSphere(pos, radius, index, matType, color, roughness, ior, att);
}

__global__ void DeviceFunc::CreateBox(Scene* scene,
									  vec3 pos,
									  vec3 length,
									  vec3 width,
									  vec3 height,
									  int index,
									  MaterialType matType,
									  vec3 color,
									  float roughness,
									  float ior,
									  vec3 att)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->CreateBox(pos, length, width, height, index, matType, color, roughness, ior, att);
}

__global__ void DeviceFunc::CreatePoly(Scene* scene,
									   vec3 pos,
									   vec3* vertices,
									   int vtxCount,
									   int index,
									   MaterialType matType,
									   vec3 color,
									   float roughness,
									   float ior,
									   vec3 att)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->CreatePoly(pos, vertices, vtxCount, index, matType, color, roughness, ior, att);
}

__global__ void DeviceFunc::SetAmbient(Scene* scene, vec3 color)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->SetAmbient(color);
}

__global__ void DeviceFunc::ClearScene(Scene* scene)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->Clear();
}

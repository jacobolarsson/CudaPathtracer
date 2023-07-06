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
									float roughness)
{
	m_objects[index] = new Sphere(pos, CreateMaterial(matType, color, roughness), radius);
}

__device__ void Scene::CreateBox(vec3 pos,
								 vec3 length,
								 vec3 width,
								 vec3 height,
								 int index,
								 MaterialType matType,
								 vec3 color,
								 float roughness)
{
	m_objects[index] = new Box(pos, CreateMaterial(matType, color, roughness), length, width, height);
}

__device__ void Scene::CreateMesh(vec3 pos,
								  vec3* vertices, 
								  int vtxCount, 
								  Face* faces,
								  int facesCount, 
								  int index, 
								  MaterialType matType, 
								  vec3 color, 
								  float roughness)
{
	m_objects[index] = new Mesh(pos, CreateMaterial(matType, color, roughness), vertices, vtxCount, faces, facesCount);
}

__device__ vec3 Scene::QueryRay(Ray const& ray, int depth, curandState* randState) const
{
	static const float cInfinity = std::numeric_limits<float>::max();

	vec3 resultColor(1.0f, 1.0f, 1.0f);
	Ray newRay = ray;

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
		// No hit
		if (hitObjIdx == -1) {
			return resultColor * m_ambient;
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
	//return 0.5f * vec3(hitData.normal.x + 1.0f, hitData.normal.y + 1.0f, hitData.normal.z + 1.0f);
}

__device__ void Raytracer::Scene::Clear()
{
	for (int i = 0; i < m_size; i++) {
		delete m_objects[i];
	}
}

__device__ Material* Scene::CreateMaterial(MaterialType type, vec3 color, float roughtness)
{
	if (type == MaterialType::LAMBERTIAN) {
		return new Lambertian(color);
	} else if (type == MaterialType::METAL) {
		return new Metal(color, roughtness);
	} else if (type == MaterialType::LIGHT) {
		return new Light(color);
	}
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
										 float roughness)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->CreateSphere(pos, radius, index, matType, color, roughness);
}

__global__ void DeviceFunc::CreateBox(Scene* scene,
									  vec3 pos,
									  vec3 length,
									  vec3 width,
									  vec3 height,
									  int index,
									  MaterialType matType,
									  vec3 color,
									  float roughness)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->CreateBox(pos, length, width, height, index, matType, color, roughness);
}

__global__ void DeviceFunc::CreateMesh(Scene* scene,
									   vec3 pos, 
									   vec3* vertices, 
									   int vtxCount, 
									   Face* faces,
									   int faceCount, 
									   int index,
									   MaterialType matType, 
									   vec3 color, 
									   float roughness)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->CreateMesh(pos, vertices, vtxCount, faces, faceCount, index, matType, color, roughness);
}

__global__ void DeviceFunc::SetAmbient(Scene* scene, vec3 color)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) {
		return;
	}
	scene->SetAmbient(color);
}

__global__ void Raytracer::DeviceFunc::ClearScene(Scene* scene)
{
	scene->Clear();
}

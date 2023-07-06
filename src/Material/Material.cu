#include "Material.h"

using namespace Raytracer;

__device__ static const float cEpsilon = 0.001f;

__device__ bool Lambertian::BounceRay(Ray const& ray,
									  HitData const& hitData,
									  Ray& newRay,
									  curandState* randState) const
{
	// Get a random vector from a sphere of radius 1

	float theta = curand_uniform(randState) * c2Pi;
	float phi = glm::acos(curand_uniform(randState) * 2.0f - 1.0f);
	vec3 randSphereVec = { glm::sin(phi) * glm::cos(theta), glm::sin(phi) * glm::sin(theta), glm::cos(phi) };

	vec3 unitSpherePoint = hitData.hitPoint + hitData.normal + randSphereVec;
	newRay.orig = hitData.hitPoint + hitData.normal * cEpsilon;
	// Get a new direction using Lambertian distribution
	newRay.dir = glm::normalize(unitSpherePoint - hitData.hitPoint);
	return true;
}

__device__ bool Metal::BounceRay(Ray const& ray,
								 HitData const& hitData,
								 Ray& newRay,
								 curandState* randState) const
{
	vec3 reflectionVec = ray.dir - hitData.normal * glm::dot(ray.dir, hitData.normal) * 2.0f;
	newRay.orig = hitData.hitPoint + glm::normalize(hitData.normal) * cEpsilon;

	// Generate a random 3D vector which coordinates are regulary distributed within
	// the volume of a ball of m_roughness radius

	vec3 ballVec(curand_uniform(randState) * 2.0f - 1.0f,
				 curand_uniform(randState) * 2.0f - 1.0f,
				 curand_uniform(randState) * 2.0f - 1.0f);
	ballVec = glm::normalize(ballVec);
	ballVec *= m_roughness * cbrtf(curand_uniform(randState));

	// Get a new direction using the reflection vector and applying the roughness sphere
	newRay.dir = glm::normalize(reflectionVec + ballVec);
	// If the new direction is pointing towards the object, no bounce
	if (glm::dot(newRay.dir, hitData.normal) < 0.0f) {
		return false;
	}
	return true;
}

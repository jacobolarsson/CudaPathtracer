#include "Material.h"

using namespace Raytracer;

__device__ static const float cEpsilon = 0.001f;

__device__ bool Lambertian::BounceRay(Ray const& ray,
									  HitData const& hitData,
									  Ray& newRay,
									  curandState* randState)
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
								 curandState* randState)
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

__device__ bool Dielectric::BounceRay(Ray const& ray, HitData const& hitData, Ray& newRay, curandState* randState)
{
	m_applyAtt = false;

	bool insideObj = glm::dot(ray.dir, hitData.normal) > 0.0f;
	vec3 norm = insideObj ? -hitData.normal : hitData.normal;

	// Compute the values to decide whether reflect, refract or not bounce the ray

	float cosTheta = glm::dot(-ray.dir, norm);
	float ni = 1.0f;
	float nt = 1.0f;

	if (insideObj) {
		ni = m_ior;
		m_timeInsideMat = hitData.t;
	}
	else {
		nt = m_ior;
	}

	float iorQuo = ni / nt;
	float eQuoSqrt = glm::sqrt(1 - iorQuo * iorQuo * (1 - cosTheta * cosTheta));

	float niCosTheta = ni * cosTheta;
	float ntCosTheta = nt * cosTheta;

	float eQuoPerp = (niCosTheta - nt * eQuoSqrt) / (niCosTheta + nt * eQuoSqrt);
	float eQuoParal = (ntCosTheta - ni * eQuoSqrt) / (ntCosTheta + ni * eQuoSqrt);

	float reflectCoeff = 0.5f * (eQuoPerp * eQuoPerp + eQuoParal * eQuoParal);
	float transCoeff = 1.0f - reflectCoeff;
	float invTransCoeff = 1.0f - transCoeff;
	float randVal = curand_uniform(randState);

	// Reflect the ray
	if (1.0f - reflectCoeff - transCoeff <= randVal && randVal < invTransCoeff) {
		newRay.orig = hitData.hitPoint + glm::normalize(norm) * cEpsilon;
		newRay.dir = glm::reflect(ray.dir, norm);

		return true;
	}
	// Refract the ray
	else if (invTransCoeff <= randVal) {
		newRay.orig = hitData.hitPoint - glm::normalize(norm) * cEpsilon;
		newRay.dir = glm::refract(ray.dir, norm, iorQuo);

		if (insideObj) {
			m_applyAtt = true;
		}
		return true;
	}
	return false;
}

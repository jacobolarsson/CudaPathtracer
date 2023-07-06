#pragma once

#include "../Math.h"

namespace Raytracer
{
	class Material
	{
	public:
		__device__ Material(vec3 color) : m_color(color) {}

		__device__ virtual bool BounceRay(Ray const&,
										  HitData const&,
										  Ray&,
										  curandState* randState) const = 0;
		__device__ vec3 GetColor() const { return m_color; }

	protected:
		vec3 m_color;
	};

	class Lambertian : public Material
	{
	public:
		__device__ Lambertian(vec3 color) : Material(color) {}

		__device__ bool BounceRay(Ray const& ray,
								  HitData const& hitData,
								  Ray& newRay,
								  curandState* randState) const override;
	};

	class Metal : public Material
	{
	public:
		__device__ Metal(vec3 color, float roughness)
			: Material(color)
			, m_roughness(roughness)
		{}

		__device__ bool BounceRay(Ray const& ray,
								  HitData const& hitData,
								  Ray& newRay,
								  curandState* randState) const override;

	private:
		float m_roughness;
	};

	class Light : public Material
	{
	public:
		__device__ Light(vec3 color) : Material(color) {}

		__device__ bool BounceRay(Ray const&,
								  HitData const&,
								  Ray&,
								  curandState*) const override
		{ return false; }
	};
}

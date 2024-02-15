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
										  curandState* randState) = 0;
		__device__ virtual vec3 GetColor() const { return m_color; }

	protected:
		vec3 m_color;
	};

	class Lambertian : public Material
	{
	public:
		__host__ __device__ Lambertian(vec3 color) : Material(color) {}

		__device__ bool BounceRay(Ray const& ray,
								  HitData const& hitData,
								  Ray& newRay,
								  curandState* randState) override;
	};

	class Metal : public Material
	{
	public:
		__host__ __device__ Metal(vec3 color, float roughness)
			: Material(color)
			, m_roughness(roughness)
		{}

		__device__ bool BounceRay(Ray const& ray,
								  HitData const& hitData,
								  Ray& newRay,
								  curandState* randState) override;

	private:
		float m_roughness;
	};

	class Dielectric : public Material
	{
	public:
		__host__ __device__ Dielectric(vec3 color, float ior, vec3 att)
			: Material(color)
			, m_ior(ior)
			, m_att(att)
			, m_applyAtt(false)
			, m_timeInsideMat(0.0f)
		{}

		__device__ bool BounceRay(Ray const& ray,
					   HitData const& hitData,
					   Ray& newRay,
					   curandState* randState) override;

		__device__ vec3 GetColor() const override
		{
			return m_applyAtt ? m_color * glm::pow(m_att, vec3(m_timeInsideMat)) : m_color;
		}

	private:
		float m_ior;
		vec3 m_att;
		bool m_applyAtt;
		float m_timeInsideMat;
	};

	class Light : public Material
	{
	public:
		__device__ Light(vec3 color) : Material(color) {}

		__device__ bool BounceRay(Ray const&,
								  HitData const&,
								  Ray&,
								  curandState*) override
		{ return false; }
	};
}

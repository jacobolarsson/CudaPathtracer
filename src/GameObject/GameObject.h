#pragma once

#include "../Math.h"
#include "../Material/Material.h"

namespace Raytracer
{

	class GameObject
	{
	public:
		__device__ GameObject(vec3 pos, Material* material)
			: m_pos(pos)
			, m_material(material)
		{}
		__device__ virtual ~GameObject() { delete m_material; }

		__device__ virtual bool RayHits(Ray const& ray, HitData& hitData, float tMax) const = 0;

		__device__ vec3 GetPos() const { return m_pos; }
		__device__ void SetPos(vec3 pos) { m_pos = pos; }
		__device__ Material* GetMaterial() const { return m_material; }

	protected:
		vec3 m_pos;
		Material* m_material;
	};

	class Sphere : public GameObject
	{
	public:
		__device__ Sphere(vec3 pos, Material* material, float radius)
			: GameObject(pos, material)
			, m_radius(radius)
		{}

		__device__ bool RayHits(Ray const& ray, HitData& hitData, float tMax) const override;

	private:
		float m_radius;
	};

	class Box : public GameObject
	{
	public:
		__device__ Box(vec3 pos, Material* material, vec3 length, vec3 width, vec3 height)
			: GameObject(pos, material)
			, m_length(length)
			, m_width(width)
			, m_height(height)
		{}

		__device__ bool RayHits(Ray const& ray, HitData& hitData, float tMax) const override;

	private:
		vec3 m_length;
		vec3 m_width;
		vec3 m_height;
	};

	struct Face
	{
		vec3 faceNormal = { 0.0f, 0.0f, 0.0f };
		int vtxIndices[3] = { -1, -1, -1 };
	};

	class Mesh : public GameObject
	{
	public:
		__device__ Mesh(vec3 pos,
						Material* material, 
						vec3* vertices, 
						int vtxCount, 
						Face* faces, 
						int faceCount)
			: GameObject(pos, material)
			, m_vertices(vertices)
			, m_vtxCount(vtxCount)
			, m_faces(faces)
			, m_faceCount(faceCount)
		{}
		__device__ ~Mesh() override
		{
			cudaFree(m_vertices);
			cudaFree(m_faces);
		}

		__device__ bool RayHits(Ray const& ray, HitData& hitData, float tMax) const override;

	private:
		__device__ float RayFaceIntersection(Ray const& ray, int faceIdx) const;

		vec3* m_vertices;
		int m_vtxCount;
		Face* m_faces;
		int m_faceCount;
	};
}

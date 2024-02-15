#pragma once

#include "../Math.h"
#include "../Material/Material.h"

namespace Raytracer
{
	struct BoundingVolume
	{
		__host__ __device__ BoundingVolume() : min{}, max{} {}
		__host__ __device__ BoundingVolume(vec3 mi, vec3 ma) : min(mi), max(ma) {}

		__host__ __device__ bool RayAABBInterval(Ray const& ray, float& tMin, float& tMax) const;
		__host__ __device__ bool TestBoundingVolume(BoundingVolume const& other) const;
		__host__ __device__ float SurfaceArea() const;
		__host__ __device__ unsigned MaxExtentDim() const;

		vec3 min{};
		vec3 max{};
	};

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
		__device__ void SetMaterial(Material* mat) { m_material	= mat; }

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
		__host__ __device__ Mesh(vec3 pos,
								 Material* material, 
								 vec3* vertices, 
								 int vtxCount, 
								 Face* faces, 
								 int faceCount,
								 BoundingVolume* faceBvs)
			: GameObject(pos, material)
			, m_vertices(vertices)
			, m_vtxCount(vtxCount)
			, m_faces(faces)
			, m_faceCount(faceCount)
			, m_boundingVolume(vec3(std::numeric_limits<float>::max()),
							   vec3(std::numeric_limits<float>::lowest()))
			, m_faceBvs(faceBvs)
		{}
		__device__ ~Mesh() override
		{
			cudaFree(m_vertices);
			cudaFree(m_faces);
			cudaFree(m_faceBvs);
		}

		__device__ bool RayHits(Ray const& ray, HitData& hitData, float tMax) const override;

		__host__ __device__ const Face* GetFaces() const { return m_faces; }
		__host__ __device__ int GetFaceCount() const { return m_faceCount; }
		__host__ __device__ BoundingVolume const& GetBoundingVolume()
		{
			for (int i = 0; i < m_vtxCount; i++) {
				m_boundingVolume.min = glm::min(m_boundingVolume.min, m_vertices[i]);
				m_boundingVolume.max = glm::max(m_boundingVolume.max, m_vertices[i]);
			}
			return m_boundingVolume;
		}
		__host__ __device__ const vec3* GetVertices() const { return m_vertices; }
		__host__ __device__ int GetVertexCount() const { return m_vtxCount; }
		__host__ __device__ const BoundingVolume* GetFaceBoundingVolumes() const { return m_faceBvs; }

	private:
		__device__ float RayFaceIntersection(Ray const& ray, int faceIdx) const;

		vec3* m_vertices;
		int m_vtxCount;
		Face* m_faces;
		int m_faceCount;
		BoundingVolume m_boundingVolume;
		BoundingVolume* m_faceBvs;
		//MaterialType m_matType;
	};

	class Polygon : public GameObject
	{
	public:
		__device__ Polygon(vec3 pos, Material* material, vec3* vertices, int vtxCount)
			: GameObject(pos, material)
			, m_vertices(vertices)
			, m_vtxCount(vtxCount)
		{}
		__device__ ~Polygon() override
		{
			cudaFree(m_vertices);
		}

		__device__  bool RayHits(Ray const& ray, HitData& hitData, float tMax) const override;

	private:
		__device__  float RayTriangleIntersection(Ray const& ray, vec3 v2, vec3 v3) const;

		vec3* m_vertices;
		int m_vtxCount;
	};
}

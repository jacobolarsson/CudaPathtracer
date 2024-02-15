#pragma once
#include "../GameObject/GameObject.h"
#include "../Common.h"

#include <vector>

namespace Raytracer
{
	struct Triangle
	{
		__device__ Triangle(vec3 v1, vec3 v2, vec3 v3, unsigned oi)
			: vertices{ v1, v2, v3 }
			, normal(glm::normalize(glm::cross(v2 - v1, v3 - v1)))
			, objIdx(oi)
		{}

		__device__ float RayIntersection(Ray const& ray) const;

		vec3 vertices[3];
		vec3 normal;
		unsigned objIdx;
	};

	class KdNode
	{
	public:
		__host__ __device__ KdNode() : splitPos(0.0f), rightChildIdx(0) {}

		// Interior node constructor
		__host__ __device__ KdNode(float splitPosition, unsigned rightChildIndex, unsigned axis)
		{
			splitPos = splitPosition;
			flags = axis;
			rightChildIdx |= (rightChildIndex << 2);
		}
		// Leaf node constructor
		__host__ __device__ KdNode(unsigned primitiveIndex, unsigned primitiveCount)
		{
			flags = 3;
			primivIdx = primitiveIndex;
			primivCount |= (primitiveCount << 2);
		}

		__device__ bool IsLeaf() const { return (flags & 0b11) == 0b11; }
		__device__ unsigned SplitAxis() const { return flags & 0b11; }
		__device__ unsigned PrimitiveCount() const { return primivCount >> 2; }
		__device__ unsigned RightChildIdx() const { return rightChildIdx >> 2; }
		__device__ float SplitPos() const { return splitPos; }
		__device__ unsigned PrimitiveIdx() const { return primivIdx; }

	private:
		union
		{
			float splitPos;
			unsigned primivIdx;
		};
		union
		{
			unsigned rightChildIdx; // 30MSB
			unsigned primivCount; // 30MSB
			unsigned flags; // 2LSB
		};
	};

	struct PendingNodes
	{
		const KdNode* node;
		float tMin, tMax;
	};

	class KdTree
	{
	public:
		KdTree()
			: m_meshes{}
			, m_sceneBv(vec3(std::numeric_limits<float>::max()),
						vec3(std::numeric_limits<float>::lowest()))
			, m_primitives{}
			, m_nodes{}
			, m_maxPrimCount(50u)
			, m_maxDepth(10u)
			, m_nextFreeNode(0u)
			, m_intersectCost(1u)
			, m_tracerseCost(80u)
			, m_splitSamples(10)
		{}

		__device__ static Mesh* RayHits(Ray const& ray,
									    HitData& hitData, 
									    float maxT, 
									    const KdNode* kdNodes, 
									    int kdNodeCount, 
									    const Triangle* primitives, 
									    int primCount, 
									    BoundingVolume const& bv,
									    Mesh* meshes);

		void AddMesh(Mesh* mesh);
		void CreateTree();
		void CreateChildTree(unsigned nodeIdx, int depth, BoundingVolume nodeBv, std::vector<std::vector<Face>> const& meshesFaces);

		std::vector<KdNode> const& GetNodes() const { return m_nodes; }
		std::vector<Triangle> const& GetPrimitives() const { return m_primitives; }
		BoundingVolume const& GetBoundingVolume() const { return m_sceneBv; }
		std::vector<Mesh*> const& GetMeshes() const { return m_meshes; }

	private:
		void IntersectingPrimitives(std::vector<std::vector<Face>>& meshesFaces, BoundingVolume const& bv) const;

		std::vector<Mesh*> m_meshes;
		BoundingVolume m_sceneBv;
		std::vector<Triangle> m_primitives;
		std::vector<KdNode> m_nodes;
		unsigned m_maxPrimCount;
		unsigned m_maxDepth;
		unsigned m_nextFreeNode;
		unsigned m_intersectCost;
		unsigned m_tracerseCost;
		unsigned m_splitSamples;
	};
}

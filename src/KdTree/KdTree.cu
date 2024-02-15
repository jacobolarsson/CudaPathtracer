#include "KdTree.h"

using namespace Raytracer;

__device__ Mesh* KdTree::RayHits(Ray const& ray,
								 HitData& hitData,
								 float maxT,
								 const KdNode* kdNodes,
								 int kdNodeCount, 
								 const Triangle* primitives, 
								 int primCount,
								 BoundingVolume const& bv,
								 Mesh* meshes)
{
	float tMin, tMax;
	// Get the ray bv intersection min and max, or return in case of no intersection
	if (kdNodes == nullptr|| !bv.RayAABBInterval(ray, tMin, tMax)) {
		return nullptr;
	}
	int finalHitObjIdx = -1;
	vec3 invRayDir(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);

	// Continer for the pending nodes
	static const int maxPendingCount = 64;
	PendingNodes pendingNodes[maxPendingCount];
	int pendingIdx = 0;

	bool hit = false;
	const KdNode* node = &kdNodes[0];
	while (node != nullptr) {
		// Return if we found a hit closer than the current node's min
		if (hitData.t < tMin) {
			break;
		}
		// If interior node
		if (!node->IsLeaf()) {
			unsigned axis = node->SplitAxis();
			// Intersection time for the split plane
			float tSplit = (node->SplitPos() - ray.orig[axis]) * invRayDir[axis];

			const KdNode* firstChild = nullptr;
			const KdNode* secondChild = nullptr;
			bool rayBehindSplitP =
				ray.orig[axis] < node->SplitPos() ||
				(ray.orig[axis] == node->SplitPos() && ray.dir[axis] <= 0);

			// Assign children nodes
			if (rayBehindSplitP) {
				firstChild = node + 1;
				secondChild = &kdNodes[node->RightChildIdx()];
			} else {
				firstChild = &kdNodes[node->RightChildIdx()];
				secondChild = node + 1;
			}
			// Process children nodes
			if (tSplit > tMax || tSplit < 0.0f) {
				node = firstChild;
			} else if (tSplit < tMin) {
				node = secondChild;
			} else { // Enqueue second child in pending node list
				pendingNodes[pendingIdx].node = secondChild;
				pendingNodes[pendingIdx].tMin = tSplit;
				pendingNodes[pendingIdx].tMax = tMax;
				pendingIdx++;
				node = firstChild;
				tMax = tSplit;
			}
		} else { // If leaf node
			unsigned primIdx = node->PrimitiveIdx();
			unsigned primCount = node->PrimitiveCount();
			int hitObjIdx = -1;
			float hitTMin = maxT;
			vec3 norm{};

			// Check for ray intersection for every primitive in the leaf node
			for (unsigned i = 0; i < primCount; i++) {
				Triangle const& trian = primitives[primIdx++];
				float hitT = trian.RayIntersection(ray);
				if (hitT < 0.0f || hitT > hitTMin) {
					continue;
				}
				hitTMin = hitT;
				hitObjIdx = trian.objIdx;
				norm = trian.normal;
			}
			if (hitObjIdx != -1) {
				hitData.t = hitTMin;
				hitData.normal = norm;
				hitData.hitPoint = ray.at(hitTMin);
				finalHitObjIdx = hitObjIdx;
			}

			// If we still have to explore pending nodes
			if (pendingIdx > 0) {
				pendingIdx--;
				node = pendingNodes[pendingIdx].node;
				tMin = pendingNodes[pendingIdx].tMin;
				tMax = pendingNodes[pendingIdx].tMax;
			} else {
				break;
			}
		}
	}
	return finalHitObjIdx == -1 ? nullptr : &meshes[finalHitObjIdx];
}

void KdTree::AddMesh(Mesh* mesh)
{
	BoundingVolume meshBv = mesh->GetBoundingVolume();
	m_sceneBv.min = glm::min(m_sceneBv.min, meshBv.min);
	m_sceneBv.max = glm::max(m_sceneBv.max, meshBv.max);

	m_meshes.push_back(mesh);
}

void KdTree::CreateTree()
{
	std::vector<std::vector<Face>> meshesFaces;
	for (Mesh* mesh : m_meshes) {
		meshesFaces.emplace_back(mesh->GetFaces(), mesh->GetFaces() + mesh->GetFaceCount());
	}
	// If there are no meshes in the scene, do not create the kdtree
	if (m_meshes.empty()) {
		return;
	}
	CreateChildTree(0, m_maxDepth, m_sceneBv, meshesFaces);
}

void KdTree::CreateChildTree(unsigned nodeIdx,
						int depth,
						BoundingVolume nodeBv,
						std::vector<std::vector<Face>> const& meshesFaces)
{
	m_nextFreeNode++;

	size_t primCount = 0;
	for (auto const& faceVec : meshesFaces) {
		primCount += faceVec.size();
	}
	// Create leaf node if criteria is met
	if (primCount < m_maxPrimCount || depth == 0) {
		// Allocate enough space for the nodes
		if (m_nodes.size() < static_cast<size_t>(nodeIdx)) {
			m_nodes.resize(static_cast<size_t>(nodeIdx));
		}
		m_nodes.emplace(m_nodes.begin() + static_cast<size_t>(nodeIdx),
						static_cast<unsigned>(m_primitives.size()), static_cast<unsigned>(primCount));

		// Copy the triangles that are inside the node bounding volume into the primitives vector
		for (size_t i = 0; i < meshesFaces.size(); i++) {
			for (auto const& face : meshesFaces.at(i)) {
				const vec3* vertices = m_meshes.at(i)->GetVertices();
				m_primitives.emplace_back(vertices[face.vtxIndices[0]],
										  vertices[face.vtxIndices[1]],
										  vertices[face.vtxIndices[2]],
										  static_cast<unsigned>(i));
			}
		}
		return;
	}

	unsigned axis = nodeBv.MaxExtentDim();

	float dimSize = nodeBv.max[axis] - nodeBv.min[axis];
	float splitDelta = dimSize / static_cast<float>(m_splitSamples);
	float splitPos = nodeBv.min[axis] + splitDelta;
	float finalSplitPos = splitPos;

	float minCost = std::numeric_limits<float>::max();

	std::vector<std::vector<Face>> finalAMeshesFaces, finalBMeshesFaces;
	BoundingVolume finalA, finalB;

	// Check the cost of splitting the node in different positions and split it in the
	// position with the least cost
	for (unsigned i = 0; i < m_splitSamples - 2; i++, splitPos += splitDelta) {
		BoundingVolume A = nodeBv;
		BoundingVolume B = nodeBv;

		std::vector<std::vector<Face>> aMeshesFaces, bMeshesFaces;

		A.max[axis] = splitPos;
		B.min[axis] = splitPos;

		IntersectingPrimitives(aMeshesFaces, A);
		IntersectingPrimitives(bMeshesFaces, B);

		size_t aPrimivCount = 0, bPrimivCount = 0;
		for (auto const& aTriangleVec : aMeshesFaces) {
			aPrimivCount += aTriangleVec.size();
		}
		for (auto const& bTriangleVec : bMeshesFaces) {
			bPrimivCount += bTriangleVec.size();
		}
		float invNodeSA = 1.0f / nodeBv.SurfaceArea();
		float currentCost = m_tracerseCost + m_intersectCost * (A.SurfaceArea() * invNodeSA * aPrimivCount +
																B.SurfaceArea() * invNodeSA * bPrimivCount);
		if (currentCost < minCost) {
			finalSplitPos = splitPos;
			minCost = currentCost;
			finalA = A;
			finalB = B;
			finalAMeshesFaces = aMeshesFaces;
			finalBMeshesFaces = bMeshesFaces;
		}
	}

	// Recursively build the children nodes and initialize this interior node

	CreateChildTree(nodeIdx + 1u, depth - 1, finalA, finalAMeshesFaces);

	m_nodes.at(static_cast<size_t>(nodeIdx)) = KdNode(finalSplitPos, m_nextFreeNode, axis);

	CreateChildTree(m_nextFreeNode, depth - 1, finalB, finalBMeshesFaces);
}

void KdTree::IntersectingPrimitives(std::vector<std::vector<Face>>& meshesFaces, BoundingVolume const& bv) const
{
	meshesFaces.clear();
	meshesFaces.resize(m_meshes.size());

	// For each mesh, find which triangles are within the nodeBv 
	for (size_t i = 0; i < m_meshes.size(); i++) {
		const Mesh* obj = m_meshes.at(i);
		for (int j = 0; j < obj->GetFaceCount(); j++) {
			// If triangle with index j from mesh with index i is inside the node bv
			if (obj->GetFaceBoundingVolumes()[j].TestBoundingVolume(bv)) {
				meshesFaces.at(i).emplace_back(obj->GetFaces()[j]);
			}
		}
	}
}

float Raytracer::Triangle::RayIntersection(Ray const& ray) const
{
	vec3 v1 = vertices[0];

	float dotValue1 = glm::dot(ray.dir, normal);
	float dotValue2 = glm::dot(ray.orig - v1, normal);

	// If no intersection
	if ((dotValue1 > 0.0f && dotValue2 > 0.0f) || (dotValue1 < 0.0f && dotValue2 < 0.0f)) {
		return -1.0f;
	}

	vec3 a = vertices[1] - v1;
	vec3 b = vertices[2] - v1;

	float t = -dotValue2 / dotValue1;
	vec3 p = ray.orig + t * ray.dir; // Ray-plane intersection point

	float sqModA = glm::dot(a, a);
	float sqModB = glm::dot(b, b);
	float dotAB = glm::dot(a, b);

	float invDet = 1.0f / (sqModA * sqModB - dotAB * dotAB);
	mat2 mtx = { { sqModB, -dotAB },
				 { -dotAB, sqModA } };

	vec3 v1P = p - v1;
	vec2 cpAB = { glm::dot(v1P, a), glm::dot(v1P, b) };

	// Scalar values for a and b to solve p = v1 + scalar1 * a + scalar2 * b
	vec2 scalarResult = invDet * mtx * cpAB;

	// If there is a ray-triangle intersection, return the time of intersection
	if (scalarResult.x >= 0.0f && scalarResult.y >= 0.0f && (scalarResult.x + scalarResult.y) <= 1.0f) {
		return t;
	}

	return -1.0f;
}

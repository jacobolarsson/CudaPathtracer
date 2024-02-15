#include "GameObject.h"

using namespace Raytracer;

__device__ void RayPlaneInterval(Ray const& ray, vec3 planeN, vec3 planeP, float& start, float& end)
{
    static const float cInfinity = std::numeric_limits<float>::max();


    float dotValue1 = glm::dot(ray.dir, planeN);
    float dotValue2 = glm::dot(ray.orig - planeP, planeN);

    // Ray is directed towards the back half-space
    if (dotValue1 <= 0.0f) {
        if (dotValue2 > 0.0f) { // Ray starts in front of the plane
            start = -dotValue2 / dotValue1;
            end = cInfinity;
        }
        else { // Ray starts in behind the plane
            start = 0.0f;
            end = cInfinity;
        }
    }
    // Ray is directed towards the front half-space
    else {
        if (dotValue2 < 0.0f) { // Ray starts in behind the plane
            start = 0.0f;
            end = -dotValue2 / dotValue1;
        }
        else { // Ray starts in front of the plane
            start = -1.0f;
            end = -1.0f;
        }
    }
}

__device__ bool Sphere::RayHits(Ray const& ray, HitData& hitData, float tMax) const
{
    vec3 oc = ray.orig - m_pos;
    float a = glm::dot(ray.dir, ray.dir);
    float b = glm::dot(oc, ray.dir);
    float c = glm::dot(oc, oc) - m_radius * m_radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - glm::sqrt(discriminant)) / a;
        if (temp < tMax && temp > 0.0f) {
            hitData.t = temp;
            hitData.hitPoint = ray.at(temp);
            hitData.normal = glm::normalize(hitData.hitPoint - m_pos);
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < tMax && temp > 0.0f) {
            hitData.t = temp;
            hitData.hitPoint = ray.at(temp);
            hitData.normal = glm::normalize(hitData.hitPoint - m_pos);
            return true;
        }
    }
    return false;
}

__device__ bool Box::RayHits(Ray const& ray, HitData& hitData, float max) const
{
    static const float cInfinity = std::numeric_limits<float>::max();
    static const float cEpsilon = 0.00001f;

    vec3 frontPlaneN = glm::normalize(glm::cross(m_length, m_height));
    vec3 leftPlaneN = glm::normalize(glm::cross(m_height, m_width));
    vec3 bottomPlaneN = glm::normalize(glm::cross(m_width, m_length));

    vec3 boxPlaneN[] = {
        frontPlaneN,     // Front
        -frontPlaneN,    // Back
        leftPlaneN,      // Left
        -leftPlaneN,     // Right
        bottomPlaneN,    // Bottom
        -bottomPlaneN    // Top
    };

    vec3 boxPlaneP[] = {
        m_pos,           // Front
        m_pos + m_width, // Back
        m_pos,           // Left
        m_pos + m_length,// Right
        m_pos,           // Bottom
        m_pos + m_height // Top
    };

    float totalTMin = 0.0f;
    float totalTMax = cInfinity;
    float minTInsersection = cInfinity;
    vec3 intersectionPlaneN;
    vec3 insideIntersectionPlaneN;

    // Get the common intersection time interval for all the planes
    for (int i = 0; i < 6; i++) {
        float tMin, tMax;
        RayPlaneInterval(ray, boxPlaneN[i], boxPlaneP[i], tMin, tMax);

        if (tMin > totalTMin) {
            totalTMin = tMin;
            intersectionPlaneN = boxPlaneN[i];
        }

        if (tMax < totalTMax) {
            totalTMax = tMax;
            insideIntersectionPlaneN = boxPlaneN[i];
        }
        // A plane interval is not valid
        if (tMin < 0.0f || tMax < 0.0f || totalTMin > totalTMax) {
            return false;
        }
    }

    // The ray is inside the box
    if (glm::abs(totalTMin) < cEpsilon) {
        if (totalTMax > max) {
            return false;
        }

        hitData.t = totalTMax;
        hitData.normal = -insideIntersectionPlaneN;
    }
    else {
        if (totalTMin > max) {
            return false;
        }

        hitData.t = totalTMin;
        hitData.normal = intersectionPlaneN;
    }

    hitData.hitPoint = ray.orig + ray.dir * hitData.t;

    return true;
}

__device__ bool Mesh::RayHits(Ray const& ray, HitData& hitData, float tMax) const
{
    float minT = std::numeric_limits<float>::max();
    int intersectionIdx = std::numeric_limits<int>::max();

    // Iterate through the mesh's triangles, check for ray intersection and
    // store the index of the triangle with the lowest intersection time
    for (int i = 0; i < m_faceCount; i++) {
        float newT = RayFaceIntersection(ray, i);

        // If the new intersection is the closest one recorded
        if (newT > 0.0f && newT < tMax && newT < minT) {
            minT = newT;
            intersectionIdx = i;
        }
    }
    // No intersection
    if (intersectionIdx >= m_faceCount) {
        return false;
    }

    hitData.hitPoint = ray.orig + minT * ray.dir;

    vec3 v1 = m_vertices[m_faces[intersectionIdx].vtxIndices[0]];
    vec3 v2 = m_vertices[m_faces[intersectionIdx].vtxIndices[1]];
    vec3 v3 = m_vertices[m_faces[intersectionIdx].vtxIndices[2]];
    vec3 norm = glm::normalize(glm::cross(v2 - v1, v3 - v1));

    hitData.normal = norm;
    hitData.t = minT;

    return true;
}

__device__ float Mesh::RayFaceIntersection(Ray const& ray, int faceIdx) const
{
    Face const& face = m_faces[faceIdx];
    vec3 v1 = m_vertices[face.vtxIndices[0]];
    vec3 planeN = face.faceNormal;

    float dotValue1 = glm::dot(ray.dir, planeN);
    float dotValue2 = glm::dot(ray.orig - v1, planeN);

    // If no intersection
    if ((dotValue1 > 0.0f && dotValue2 > 0.0f) || (dotValue1 < 0.0f && dotValue2 < 0.0f)) {
        return -1.0f;
    }

    vec3 a = m_vertices[face.vtxIndices[1]] - v1;
    vec3 b = m_vertices[face.vtxIndices[2]] - v1;

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

__device__ bool Polygon::RayHits(Ray const& ray, HitData& hitData, float tMax) const
{
    vec3 v1 = m_vertices[0];
    for (int i = 1; i < m_vtxCount - 1; i++) {
        vec3 v2 = m_vertices[i];
        vec3 v3 = m_vertices[i + 1];

        float t = RayTriangleIntersection(ray, v2, v3);
        if (t > 0.0f && t < tMax) {
            hitData.hitPoint = ray.at(t);
            hitData.normal = glm::normalize(glm::cross(v2 - v1, v3 - v1));
            hitData.t = t;
            return true;
        }
    }
    return false;
}

__device__ float Polygon::RayTriangleIntersection(Ray const& ray, vec3 v2, vec3 v3) const
{
    vec3 v1 = m_vertices[0];

    vec3 planeN = glm::normalize(glm::cross(v2 - v1, v3 - v1));

    float dotValue1 = glm::dot(ray.dir, planeN);
    float dotValue2 = glm::dot(ray.orig - v1, planeN);

    // If no intersection
    if ((dotValue1 > 0.0f && dotValue2 > 0.0f) || (dotValue1 < 0.0f && dotValue2 < 0.0f)) {
        return -1.0f;
    }
    vec3 a = v2 - v1;
    vec3 b = v3 - v1;

    float t = -dotValue2 / dotValue1;
    vec3 p = ray.at(t); // Ray-plane intersection point

    float sqModA = glm::dot(a, a);
    float sqModB = glm::dot(b, b);
    float dotAB = glm::dot(a, b);

    float invDet = 1.0f / (sqModA * sqModB - dotAB * dotAB);
    mat2 mtx = { { sqModB, -dotAB },
                 { -dotAB, sqModA } };

    vec3 v1P = p - v1;
    vec2 cpAB = { glm::dot(v1P, a), glm::dot(v1P, b) };

    // Scalar values for a and b to solve p = v1 + scalar1 * a + scalar2 * b
    vec2 scalarResult = (invDet * mtx) * cpAB;

    // If there is a ray-triangle intersection, return the time of intersection
    if (scalarResult.x >= 0.0f && scalarResult.y >= 0.0f && (scalarResult.x + scalarResult.y) <= 1.0f) {
        return t;
    }

    return -1.0f;
}

__device__ bool BoundingVolume::RayAABBInterval(Ray const& ray, float& tMin, float& tMax) const
{
    tMin = 0.0f;
    tMax = std::numeric_limits<float>::max();

    for (int axis = 0; axis < 3; ++axis) {
        float invD = 1.0f / ray.dir[axis];
        float t0 = (min[axis] - ray.orig[axis]) * invD;
        float t1 = (max[axis] - ray.orig[axis]) * invD;
        if (invD < 0.0f) {
            float temp = t1;
            t1 = t0;
            t0 = temp;
        }

        tMin = glm::max(tMin, t0);
        tMax = glm::min(tMax, t1);

        if (tMax <= tMin) {
            return false;
        }
    }
    return true;
}

bool BoundingVolume::TestBoundingVolume(BoundingVolume const& other) const
{
    return min.x <= other.max.x &&
           max.x >= other.min.x &&
           min.y <= other.max.y &&
           max.y >= other.min.y &&
           min.z <= other.max.z &&
           max.z >= other.min.z;
}

float BoundingVolume::SurfaceArea() const
{
    vec3 diag = max - min;
    return 2.0f * (diag.x * diag.y + diag.x * diag.z + diag.y * diag.z);
}

unsigned BoundingVolume::MaxExtentDim() const
{
    vec3 diag = max - min;
    if (diag.x > diag.y && diag.x > diag.z) {
        return 0u;
    }
    else if (diag.y > diag.z) {
        return 1u;
    }
    else {
        return 2u;
    }
}

#pragma once

#include "Common.h"

#include <numeric>

__device__ static const float c2Pi = 6.28318530717958647692528676655900576f;

namespace Raytracer
{
    struct HitData
    {
        vec3 hitPoint{};
        vec3 normal{};
        float t = 0.0f;
    };

    struct Ray
    {
        __device__ Ray(vec3 o, vec3 d)
            : orig(o)
            , dir(d)
        {}

        __device__ vec3 at(float t) const
        {
            return orig + t * dir;
        }

        vec3 orig{};
        vec3 dir{};
    };
}
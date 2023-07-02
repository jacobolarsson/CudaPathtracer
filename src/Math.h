#ifndef MATH_H
#define MATH_H

#include "Common.h"

#include <math.h>

namespace Raytracer
{
    #define MAX(x, y) (((x) > (y)) ? (x) : (y))
    #define MIN(x, y) (((x) < (y)) ? (x) : (y))

    struct vec3
    {
        __host__ __device__ vec3()
            : x(0.0f)
            , y(0.0f)
            , z(0.0f)
        {}

        __host__ __device__ vec3(float e0, float e1, float e2)
            : x(e0)
            , y(e1)
            , z(e2)
        {}

        __host__ __device__ float length() const {
            return sqrt(length_squared());
        }

        __host__ __device__ float length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        __host__ __device__ vec3& operator/=(float rhs) {
            e[0] /= rhs;
            e[1] /= rhs;
            e[2] /= rhs;
            return *this;
        }

        __host__ __device__ void normalize() {
            *this /= length();
        }
        
        union
        {
            struct
            {
                float x, y, z;
            };
            float e[3];
        };
    };

    __host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
        return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
    }

    __host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
        return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
    }

    __host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
        return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
    }

    __host__ __device__ inline vec3 operator*(float t, const vec3& v) {
        return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    __host__ __device__ inline vec3 operator*(const vec3& v, float t) {
        return t * v;
    }

    __host__ __device__ inline vec3 operator/(vec3 v, float t) {
        return (1.0f / t) * v;
    }

    __host__ __device__ inline float dot(const vec3& u, const vec3& v) {
        return u.e[0] * v.e[0]
               + u.e[1] * v.e[1]
               + u.e[2] * v.e[2];
    }

    __host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
        return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                    u.e[2] * v.e[0] - u.e[0] * v.e[2],
                    u.e[0] * v.e[1] - u.e[1] * v.e[0]);
    }

    __host__ __device__ inline vec3& normalize(vec3& v) {
        v.normalize();
        return v;
    }

    __host__ __device__ inline float clamp(float minVal, float maxVal, float val) {
        return MIN(maxVal, MAX(val, minVal));
    }

    struct ray
    {
        __device__ ray() {}
        __device__ ray(vec3 o, vec3 d)
            : orig(o)
            , dir(d)
        {}

        __device__ vec3 at(float t)
        {
            return orig + t * dir;
        }

        vec3 orig;
        vec3 dir;
    };
}

#endif
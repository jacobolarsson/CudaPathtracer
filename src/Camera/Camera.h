#pragma once

#include "../Math.h"

namespace Raytracer
{
	struct Camera
	{
		Camera(vec3 p, vec3 t, vec3 u, float f)
			: pos(p)
			, target(t)
			, up(u)
			, focal(f)
		{}

		vec3 pos;
		vec3 target;
		vec3 up;
		float focal;
	};
}

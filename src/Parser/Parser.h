#pragma once

#include "../Scene/Scene.h"
#include "../Camera/Camera.h"

#include <string>
#include <memory>

namespace Raytracer
{
	class Parser
	{
	public:
		Parser(std::string const& filename, Scene* scene)
			: m_filename(filename)
			, m_dScene(scene)
		{}
		void LoadScene();
		int GetObjectCount() const;
		std::unique_ptr<Camera>& GetCamera() { return m_camera; }

	private:
		void SetVertexData(std::string const& filename,
					  vec3 pos,
					  vec3 orientation, 
					  float scale, 
					  vec3*& vertices, 
					  int& vertexCount, 
					  Face*& faces, 
					  int& faceCount);

		std::string m_filename;
		Scene* m_dScene;
		std::unique_ptr<Camera> m_camera;
	};
}

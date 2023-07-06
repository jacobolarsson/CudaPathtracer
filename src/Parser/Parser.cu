#define TINYOBJLOADER_IMPLEMENTATION

#include "Parser.h"

#include <fstream>
#include <exception>
#include "../../include/tiny_obj_loader.h"
#include "../../include/glm/gtc/matrix_transform.hpp"
#include "../../include/glm/gtx/euler_angles.hpp"

using namespace Raytracer;

int ReadInt(std::ifstream& file)
{
	std::string str;
	file >> str;
	return static_cast<int>(std::atoi(str.c_str()));
}

float ReadFloat(std::ifstream& file)
{
	std::string str;
	file >> str;
	return static_cast<float>(std::atof(str.c_str()));
}

vec3 ReadVec3(std::ifstream& file)
{
	vec3 vec;
	std::string str;
	file >> str;

	size_t offset = 1;
	auto commaPos = str.find_first_of(',', offset);

	// Read first float of the vector
	std::string floatString(str.begin() + offset, str.begin() + commaPos);
	vec.x = static_cast<float>(std::atof(floatString.c_str()));

	offset = commaPos + 1;
	commaPos = str.find_first_of(',', offset);

	// Read the second float of the vector
	floatString = std::string(str.begin() + offset, str.begin() + commaPos);
	vec.y = static_cast<float>(std::atof(floatString.c_str()));

	offset = commaPos + 1;

	// Read the third float of the vector
	floatString = std::string(str.begin() + offset, str.end());
	vec.z = static_cast<float>(std::atof(floatString.c_str()));

	return vec;
}

void Parser::LoadScene()
{
	std::ifstream file(m_filename);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open scene file");
	}

	int count = GetObjectCount();
	std::string str{};
	int index = 0;

	// Load the game object, camera and lighthing data from the scene file
	while (!file.eof()) {
		str = "";
		file >> str;

		bool comment = str.find_first_of('#') == 0;
		if (comment) {
			std::getline(file, str);
			continue;
		}
		// Load sphere objects' data
		if (str == "SPHERE") {
			vec3 pos = ReadVec3(file);
			float radius = ReadFloat(file);

			std::getline(file, str);
			file >> str;

			if (str == "DIFFUSE") {
				vec3 color = ReadVec3(file);
				DeviceFunc::CreateSphere<<<1, 1>>>(m_dScene, pos, radius, index++, MaterialType::LAMBERTIAN, color);
			}
			else if (str == "METAL") {
				vec3 color = ReadVec3(file);
				float roughness = ReadFloat(file);
				DeviceFunc::CreateSphere<<<1, 1>>>(m_dScene, pos, radius, index++, MaterialType::METAL, color, roughness);
			}
		}
		// Load box objects' data
		else if (str == "BOX") {
			vec3 pos = ReadVec3(file);
			vec3 length = ReadVec3(file);
			vec3 width = ReadVec3(file);
			vec3 height = ReadVec3(file);
			
			std::getline(file, str);
			file >> str;

			if (str == "DIFFUSE") {
				vec3 color = ReadVec3(file);
				DeviceFunc::CreateBox<<<1, 1>>>(m_dScene, pos, length, width, height, index++, MaterialType::LAMBERTIAN, color);
			}
			if (str == "METAL") {
				vec3 color = ReadVec3(file);
				float roughness = ReadFloat(file);
				DeviceFunc::CreateBox<<<1, 1>>>(m_dScene, pos, length, width, height, index++, MaterialType::METAL, color, roughness);
			}
		}
		// Load mesh objects' data
		else if (str == "MESH") {
			std::string filename;
			file >> filename;
			vec3 pos = ReadVec3(file);
			vec3 orientation = glm::radians(ReadVec3(file));
			float scale = ReadFloat(file);

			vec3* vertices;
			Face* faces;
			int vtxCount, faceCount;
			SetVertexData(filename, pos, orientation, scale, vertices, vtxCount, faces, faceCount);

			vec3* dVertices;
			Face* dFaces;

			//checkCudaErrors(cudaMalloc(&dVertices, vtxCount * sizeof(vec3)));
			//checkCudaErrors(cudaMalloc(&dFaces, faceCount * sizeof(Face)));

			//checkCudaErrors(cudaMemcpy(dVertices, vertices, vtxCount * sizeof(vec3), cudaMemcpyHostToDevice));
			//checkCudaErrors(cudaMemcpy(dFaces, faces, faceCount * sizeof(Face), cudaMemcpyHostToDevice));

			std::getline(file, str);
			file >> str;

			if (str == "DIFFUSE") {
				vec3 color = ReadVec3(file);
				DeviceFunc::CreateMesh<<<1, 1>>>(m_dScene, pos, vertices, vtxCount, faces, faceCount, index++, MaterialType::LAMBERTIAN, color);
			}
			if (str == "METAL") {
				vec3 color = ReadVec3(file);
				float roughness = ReadFloat(file);
				DeviceFunc::CreateMesh<<<1, 1>>>(m_dScene, pos, vertices, vtxCount, faces, faceCount, index++, MaterialType::METAL, color, roughness);
			}
		}

		// Load light source data
		else if (str == "LIGHT") {
			vec3 pos = ReadVec3(file);
			float radius = ReadFloat(file);
			vec3 color = ReadVec3(file);
			DeviceFunc::CreateSphere<<<1, 1>>>(m_dScene, pos, radius, index++, MaterialType::LIGHT, color);
		}

		// Load ambient light
		else if (str == "AMBIENT") {
			DeviceFunc::SetAmbient<<<1, 1>>>(m_dScene, ReadVec3(file));
		}
		// Load camera data
		else if (str == "CAMERA") {
			vec3 pos = ReadVec3(file);
			vec3 target = ReadVec3(file);
			vec3 up = ReadVec3(file);
			float focalLength = ReadFloat(file);

			m_camera = std::make_unique<Camera>(pos, target, up, focalLength);
		}
	}
}

int Parser::GetObjectCount() const
{
	std::ifstream file(m_filename);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open scene file");
	}

	int count = 0;
	std::string str{};
	// Count how many objects are in the scene file
	while (!file.eof()) {
		str = "";
		file >> str;
		if (str == "DIFFUSE" || str == "LIGHT" || str == "METAL" || str == "DIELECTRIC") {
			count++;
		}
		std::getline(file, str);
	}
	return count;
}

void Parser::SetVertexData(std::string const& filename,
						   vec3 pos,
						   vec3 orientation, 
						   float scale, 
						   vec3*& vertices, 
						   int& vertexCount, 
						   Face*& faces, 
						   int& faceCount)
{
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename)) {
		if (!reader.Error().empty()) {
			throw std::runtime_error("TinyObjReader: " + reader.Error());
		}
		exit(1);
	}
	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning() << std::endl;
	}

	tinyobj::attrib_t const& attrib = reader.GetAttrib();
	std::vector<tinyobj::shape_t> const& shapes = reader.GetShapes();

	vertexCount = attrib.vertices.size() / 3;
	faceCount = shapes.at(0).mesh.num_face_vertices.size();

	// Allocate unified memory for the vertex data
	checkCudaErrors(cudaMallocManaged(&vertices, vertexCount * sizeof(vec3)));
	checkCudaErrors(cudaMallocManaged(&faces, faceCount * sizeof(Face)));

	//checkCudaErrors(cudaMallocHost(&vertices, vertexCount * sizeof(vec3)));
	//checkCudaErrors(cudaMallocHost(&faces, faceCount * sizeof(Face)));

	// Copy the mesh vertices
	for (size_t i = 0, j = 0; i < attrib.vertices.size(); i += 3, j++) {
		vertices[j] = { attrib.vertices.at(i), attrib.vertices.at(i + 1), attrib.vertices.at(i + 2) };
	}

	size_t indexOff = 0;
	// Loop over faces(triangle)
	for (size_t i = 0; i < shapes.at(0).mesh.num_face_vertices.size(); i++) {
		faces[i].vtxIndices[0] = shapes.at(0).mesh.indices[indexOff].vertex_index;
		faces[i].vtxIndices[1] = shapes.at(0).mesh.indices[indexOff + 1].vertex_index;
		faces[i].vtxIndices[2] = shapes.at(0).mesh.indices[indexOff + 2].vertex_index;

		indexOff += 3;
	}
	// Transform every vertex of the mesh from model to world coordinate system
	for (int i = 0; i < vertexCount; i++) {
		glm::mat4 modelToWorld = glm::translate(mat4(1.0f), pos) *
								 glm::eulerAngleZYX(orientation.x, orientation.y, orientation.z) *
								 glm::scale(mat4(1.0f), vec3(scale));
		vertices[i] = modelToWorld * vec4(vertices[i], 1.0f);
	}

	// Once we have all the vertices in world space, compute the face normals
	for (int i = 0; i < faceCount; i++) {
		vec3 v0 = vertices[faces[i].vtxIndices[0]];
		vec3 v1 = vertices[faces[i].vtxIndices[1]];
		vec3 v2 = vertices[faces[i].vtxIndices[2]];

		faces[i].faceNormal = normalize(cross(v1 - v0, v2 - v0));
	}
}

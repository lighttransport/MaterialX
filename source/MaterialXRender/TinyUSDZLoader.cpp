//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/TinyUSDZLoader.h>

#if defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wswitch"
#endif

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

// <MaterialX>/third_party/tinyusdz/src
#include <tinyusdz.hh>
#include <io-util.hh>
#include <pprinter.hh>
#include <value-pprint.hh>
#include <tydra/render-data.hh>

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#if defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <cstring>
#include <iostream>
#include <limits>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const float MAX_FLOAT = std::numeric_limits<float>::max();
const size_t FACE_VERTEX_COUNT = 3;

// List of transforms which match to meshes
using USDMeshMatrixList = std::unordered_map<tinyusdz::tydra::RenderMesh*, std::vector<Matrix44>>;

#if 0
// Compute matrices for each mesh. Appends a transform for each transform instance
void computeMeshMatrices(USDMeshMatrixList& meshMatrices, tinyusdz::tydra::Node* cnode)
{
    cgltf_mesh* cmesh = cnode->mesh;
    if (cmesh)
    {
        float t[16];
        cgltf_node_transform_world(cnode, t);
        Matrix44 positionMatrix = Matrix44(
            (float) t[0], (float) t[1], (float) t[2], (float) t[3],
            (float) t[4], (float) t[5], (float) t[6], (float) t[7],
            (float) t[8], (float) t[9], (float) t[10], (float) t[11],
            (float) t[12], (float) t[13], (float) t[14], (float) t[15]);
        meshMatrices[cmesh].push_back(positionMatrix);
    }

    // Iterate over all children. Note that the existence of a mesh
    // does not imply that this is a leaf node so traversal should
    // continue even when a mesh is encountered.
    for (cgltf_size i = 0; i < cnode->children_count; i++)
    {
        computeMeshMatrices(meshMatrices, cnode->children[i]);
    }
}
#endif

// List of path names which match to meshes
using USDMeshPathList = std::unordered_map<tinyusdz::tydra::RenderMesh*, StringVec>;

#if 0 // TODO
void decodeVec4Tangents(MeshStreamPtr vec4TangentStream, MeshStreamPtr normalStream, MeshStreamPtr& tangentStream, MeshStreamPtr& bitangentStream)
{
    if (vec4TangentStream->getSize() != normalStream->getSize())
    {
        return;
    }

    tangentStream = MeshStream::create("i_" + MeshStream::TANGENT_ATTRIBUTE, MeshStream::TANGENT_ATTRIBUTE, 0);
    bitangentStream = MeshStream::create("i_" + MeshStream::BITANGENT_ATTRIBUTE, MeshStream::BITANGENT_ATTRIBUTE, 0);

    tangentStream->resize(vec4TangentStream->getSize());
    bitangentStream->resize(vec4TangentStream->getSize());

    for (size_t i = 0; i < vec4TangentStream->getSize(); i++)
    {
        const Vector4& vec4Tangent = vec4TangentStream->getElement<Vector4>(i);
        const Vector3& normal = normalStream->getElement<Vector3>(i);

        Vector3& tangent = tangentStream->getElement<Vector3>(i);
        Vector3& bitangent = bitangentStream->getElement<Vector3>(i);

        tangent = Vector3(vec4Tangent[0], vec4Tangent[1], vec4Tangent[2]);
        bitangent = normal.cross(tangent) * vec4Tangent[3];
    }
}
#endif

bool setupMeshNode(const tinyusdz::tydra::RenderScene &rscene, const tinyusdz::tydra::Node &node, MeshList &meshList, bool texcoordVerticalFlip, int debugLevel)
{
  // Skip non-mesh node
  if (node.nodeType != tinyusdz::tydra::NodeType::Mesh) {
    return true;
  }

  if (node.id < 0) {
    return true;
  }

	// tinyusdz::value::matrix4d uses the same memory layout of OpenGL.
	tinyusdz::value::matrix4d globalMatrix = node.global_matrix;
	//tinyusdz::value::matrix4d normalMatrix = tinyusdz::transpose(tinyusdz::inverse(globalMatrix));

	Matrix44 positionMatrix;
	{
		const double *t = &globalMatrix.m[0][0];
		positionMatrix = Matrix44(
		(float) t[0], (float) t[1], (float) t[2], (float) t[3],
		(float) t[4], (float) t[5], (float) t[6], (float) t[7],
		(float) t[8], (float) t[9], (float) t[10], (float) t[11],
		(float) t[12], (float) t[13], (float) t[14], (float) t[15]);
	}
    const Matrix44 normalMatrix = positionMatrix.getInverse().getTranspose();

	// TODO: GeomSubset

	Vector3 boxMin = { MAX_FLOAT, MAX_FLOAT, MAX_FLOAT };
	Vector3 boxMax = { -MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT };


	string meshName = node.abs_path; // NOTE: Absolute Prim path is unique
    MeshPtr mesh = Mesh::create(meshName);
	if (debugLevel > 0)
	{
		std::cout << "Translate mesh: " << meshName << std::endl;
	}
    meshList.push_back(mesh);
    //mesh->setSourceUri(filePath);

    MeshStreamPtr positionStream = nullptr;
    MeshStreamPtr normalStream = nullptr;
    MeshStreamPtr colorStream = nullptr;
    MeshStreamPtr texcoordStream = nullptr;
    MeshStreamPtr vec4TangentStream = nullptr;

	const tinyusdz::tydra::RenderMesh &rmesh = rscene.meshes[node.id];

    // Read in vertex streams
    // Position
    {
		const size_t streamIndex = 0; // 0 for now.

		if (rmesh.points.size() < 3) {
			if (debugLevel > 0)
			{
				std::cout << "Insufficient number of Mesh's points: " << rmesh.points.size() << std::endl;
			}
			return false;
		}

		{
            positionStream = MeshStream::create("i_" + MeshStream::POSITION_ATTRIBUTE, MeshStream::POSITION_ATTRIBUTE, streamIndex);
            mesh->addStream(positionStream);

            positionStream->reserve(rmesh.points.size());
            MeshFloatBuffer& buffer = positionStream->getData();

			for (size_t i = 0; i < rmesh.points.size(); i++) {
				Vector3 position;
				position[0] = rmesh.points[i][0];
				position[1] = rmesh.points[i][1];
				position[2] = rmesh.points[i][2];

				position = positionMatrix.transformPoint(position);

				buffer.push_back(position[0]);
				buffer.push_back(position[1]);
				buffer.push_back(position[2]);

				boxMin[0] = std::min(position[0], boxMin[0]);
				boxMin[1] = std::min(position[1], boxMin[1]);
				boxMin[2] = std::min(position[2], boxMin[2]);

				boxMax[0] = std::max(position[0], boxMax[0]);
				boxMax[1] = std::max(position[1], boxMax[1]);
				boxMax[2] = std::max(position[2], boxMax[2]);
			}

		}

		if ((rmesh.normals.vertex_count() > 0) && (rmesh.normals.is_vertex()) && (rmesh.normals.format == tinyusdz::tydra::VertexAttributeFormat::Vec3)) {
            normalStream = MeshStream::create("i_" + MeshStream::NORMAL_ATTRIBUTE, MeshStream::NORMAL_ATTRIBUTE, streamIndex);
            mesh->addStream(normalStream);
            normalStream->reserve(rmesh.normals.vertex_count());
            MeshFloatBuffer& buffer = normalStream->getData();
			const float *input = reinterpret_cast<const float *>(rmesh.normals.buffer());
			for (size_t i = 0; i < rmesh.normals.vertex_count(); i++) {
				Vector3 normal;
				normal[0] = input[3 * i+ 0];
				normal[1] = input[3 * i+ 1];
				normal[2] = input[3 * i+ 2];
                normal = normalMatrix.transformVector(normal).getNormalized();

                buffer.push_back(normal[0]);
                buffer.push_back(normal[1]);
                buffer.push_back(normal[2]);
			}
        }

#if 0 // TODO: tangent
        {
            vec4TangentStream = MeshStream::create("i_" + MeshStream::TANGENT_ATTRIBUTE + "4", MeshStream::TANGENT_ATTRIBUTE, streamIndex);
            vec4TangentStream->setStride(MeshStream::STRIDE_4D); // 4th component = tangent sign.
        }
#endif

		if ((rmesh.vertex_colors.vertex_count() > 0) && (rmesh.vertex_colors.is_vertex()) && (rmesh.vertex_colors.format == tinyusdz::tydra::VertexAttributeFormat::Vec3)) {
            colorStream = MeshStream::create("i_" + MeshStream::COLOR_ATTRIBUTE + "_" + std::to_string(0), MeshStream::COLOR_ATTRIBUTE, streamIndex);
            mesh->addStream(colorStream);

			bool has_opacity = (rmesh.vertex_colors.vertex_count() == rmesh.vertex_opacities.vertex_count()) && (rmesh.vertex_opacities.vertex_count() > 0) && (rmesh.vertex_opacities.is_vertex()) && (rmesh.vertex_opacities.format == tinyusdz::tydra::VertexAttributeFormat::Float);
			
			if (has_opacity) {
				colorStream->setStride(MeshStream::STRIDE_4D);
			}

			colorStream->reserve(rmesh.vertex_colors.vertex_count());
			MeshFloatBuffer& buffer = colorStream->getData();

			const float *input = reinterpret_cast<const float *>(rmesh.vertex_colors.buffer());
			const float *opacity_input = reinterpret_cast<const float *>(rmesh.vertex_opacities.buffer());

			for (size_t i = 0; i < rmesh.vertex_colors.vertex_count(); i++) {
				float col[3];
				col[0] = input[3 * i + 0];
				col[1] = input[3 * i + 1];
				col[2] = input[3 * i + 2];

				buffer.push_back(col[0]);
				buffer.push_back(col[1]);
				buffer.push_back(col[2]);
				if (has_opacity && opacity_input) {
					buffer.push_back(opacity_input[i]);
				}
			}
        }

		// texcoord0
		if (rmesh.texcoords.count(0)) {
			if ((rmesh.texcoords.at(0).vertex_count() > 0) && (rmesh.texcoords.at(0).is_vertex()) && (rmesh.texcoords.at(0).format == tinyusdz::tydra::VertexAttributeFormat::Vec2)) {
				texcoordStream = MeshStream::create("i_" + MeshStream::TEXCOORD_ATTRIBUTE + "_0", MeshStream::TEXCOORD_ATTRIBUTE, 0);
				mesh->addStream(texcoordStream);
				texcoordStream->setStride(MeshStream::STRIDE_2D);

				texcoordStream->reserve(rmesh.texcoords.at(0).vertex_count());
				MeshFloatBuffer& buffer = texcoordStream->getData();
				const float *input = reinterpret_cast<const float *>(rmesh.texcoords.at(0).buffer());

				for (size_t i = 0; i < rmesh.texcoords.at(0).vertex_count(); i++) {
					float uv[2];
					uv[0] = input[2 * i + 0];
					uv[1] = input[2 * i + 1];
					if (texcoordVerticalFlip) {
						uv[1] = 1.0f - uv[1];
					}
					buffer.push_back(uv[0]);
					buffer.push_back(uv[1]);
				}
			}
		}
	}



    // Read indexing
    MeshPartitionPtr part = MeshPartition::create();
    size_t indexCount = 0;
	indexCount = rmesh.faceVertexCounts().size(); 

	// faces are all triangulated.
    size_t faceCount = indexCount / FACE_VERTEX_COUNT;
    part->setFaceCount(faceCount);
    part->setName(meshName);

    MeshIndexBuffer& indices = part->getIndices();
    if (debugLevel > 0)
    {
        std::cout << "** Read indexing: Count = " << std::to_string(indexCount) << std::endl;
    }
	for (size_t i = 0; i < rmesh.faceVertexIndices().size(); i++) {
		indices.push_back(rmesh.faceVertexIndices()[i]);
	}

    mesh->addPartition(part);

    // Update positional information.
    mesh->setVertexCount(positionStream->getData().size() / MeshStream::STRIDE_3D);
    mesh->setMinimumBounds(boxMin);
    mesh->setMaximumBounds(boxMax);
    Vector3 sphereCenter = (boxMax + boxMin) * 0.5;
    mesh->setSphereCenter(sphereCenter);
    mesh->setSphereRadius((sphereCenter - boxMin).getMagnitude());

    if (vec4TangentStream && normalStream)
    {
        // Decode tangents primvar to MaterialX vec3 tangents and bitangents
        MeshStreamPtr tangentStream;
        MeshStreamPtr bitangentStream;
        //decodeVec4Tangents(vec4TangentStream, normalStream, tangentStream, bitangentStream);

        if (tangentStream)
        {
            mesh->addStream(tangentStream);
        }
        if (bitangentStream)
        {
            mesh->addStream(bitangentStream);
        }
    }

    // Generate tangents, normals and texture coordinates if none are provided
    if (!normalStream)
    {
        normalStream = mesh->generateNormals(positionStream);
        mesh->addStream(normalStream);
    }
    if (!texcoordStream)
    {
        texcoordStream = mesh->generateTextureCoordinates(positionStream);
        mesh->addStream(texcoordStream);
    }
    if (!vec4TangentStream)
    {
        MeshStreamPtr tangentStream = mesh->generateTangents(positionStream, normalStream, texcoordStream);
        if (tangentStream)
        {
            mesh->addStream(tangentStream);
        }
        MeshStreamPtr bitangentStream = mesh->generateBitangents(normalStream, tangentStream);
        if (bitangentStream)
        {
            mesh->addStream(bitangentStream);
        }
    }
	

	return true;
}

bool traverseNodeRecursive(const tinyusdz::tydra::RenderScene &rscene, const tinyusdz::tydra::Node &node, MeshList &meshList, bool texcoordVerticalFlip, int debugLevel)
{
	if (!setupMeshNode(rscene, node, meshList, texcoordVerticalFlip, debugLevel)) {
		return false;
	}

	for (const auto &child : node.children) {
		if (!traverseNodeRecursive(rscene, child, meshList, texcoordVerticalFlip, debugLevel)) {
			return false;
		}
	}

	return true;
}

} // anonymous namespace

bool TinyUSDZLoader::load(const FilePath& filePath, MeshList& meshList, bool texcoordVerticalFlip)
{
    const string input_filename = filePath.asString();

    if (!tinyusdz::io::FileExists(input_filename)) {
        std::cerr << "File not found: " << input_filename << "\n";
    }

    if (!tinyusdz::IsUSD(input_filename)) {
        std::cerr << "File is not a USD(USDA/USDC/USDZ) format: " << input_filename << "\n";
    }

    tinyusdz::Stage stage;
    std::string warn, err;

    bool result = tinyusdz::LoadUSDFromFile(input_filename, &stage, &warn, &err);
    if (warn.size()) {
        std::cout << "WARN: " << warn << "\n";
        warn.clear();
    }

    if (!result) {
        std::cerr << "USD Load ERROR: " << err << "\n";
        return false;
    }

    // Convert USD Scene(Stage) to OpenGL/Vulkan-friendly scene data using TinyUSDZ Tydra
    tinyusdz::tydra::RenderScene render_scene;
    tinyusdz::tydra::RenderSceneConverter converter;
    tinyusdz::tydra::RenderSceneConverterEnv env(stage);

    // In default, RenderSceneConverter triangulate meshes and build single vertex ind  ex.
    // You can explicitly enable triangulation and vertex-indices build by
    //env.mesh_config.triangulate = true;
    //env.mesh_config.build_vertex_indices = true;

    // Load textures as stored representaion(e.g. 8bit sRGB texture is read as 8bit sR  GB)
    env.material_config.linearize_color_space = false;
    env.material_config.preserve_texel_bitdepth = true;

    std::string usd_basedir = tinyusdz::io::GetBaseDir(input_filename);
    bool is_usdz = tinyusdz::IsUSDZ(input_filename);
    tinyusdz::USDZAsset usdz_asset;
	if (is_usdz) {
		// Setup AssetResolutionResolver to read a asset(file) from memory.
		if (!tinyusdz::ReadUSDZAssetInfoFromFile(input_filename, &usdz_asset, &warn, &err )) {
			std::cerr << "Failed to read USDZ assetInfo from file: " << err << "\n";
			return false;
		}
		if (warn.size()) {
			std::cout << warn << "\n";
		}

		tinyusdz::AssetResolutionResolver arr;

		// NOTE: Pointer address of usdz_asset must be valid until the call of RenderSceneConverter::ConvertToRenderScene.
		if (!tinyusdz::SetupUSDZAssetResolution(arr, &usdz_asset)) {
			std::cerr << "Failed to setup AssetResolution for USDZ asset\n";
			return false;
		};

		env.asset_resolver = arr;

	} else {
	  env.set_search_paths({usd_basedir});
	}
    

    env.timecode = tinyusdz::value::TimeCode::Default();
    bool ret = converter.ConvertToRenderScene(env, &render_scene);
    if (!ret) {
        std::cerr << "Failed to convert USD Stage to RenderScene: \n" << converter.GetError() << "\n";
        return false;
    }

    if (converter.GetWarning().size()) {
        std::cout << "ConvertToRenderScene warn: " << converter.GetWarning() << "\n";
    }

#if 0
    // Precompute mesh / matrix associations starting from the root
    // of the scene.
    USDMeshMatrixList gltfMeshMatrixList;
    for (cgltf_size sceneIndex = 0; sceneIndex < data->scenes_count; ++sceneIndex)
    {
        cgltf_scene* scene = &data->scenes[sceneIndex];
        for (cgltf_size nodeIndex = 0; nodeIndex < scene->nodes_count; ++nodeIndex)
        {
            cgltf_node* cnode = scene->nodes[nodeIndex];
            if (!cnode)
            {
                continue;
            }
            computeMeshMatrices(gltfMeshMatrixList, cnode);
        }
    }
#endif

	const tinyusdz::tydra::Node &root_node = render_scene.nodes[render_scene.default_root_node];
	if (!traverseNodeRecursive(render_scene, root_node, meshList, texcoordVerticalFlip, _debugLevel)) {
        std::cerr << "Failed to setup Mesh\n";
		return false;
	}

    return true;
}

MATERIALX_NAMESPACE_END

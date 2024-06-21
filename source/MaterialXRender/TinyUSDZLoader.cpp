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

    // Read in all meshes
    StringSet meshNames;
    for (size_t m = 0; m < data->meshes_count; m++)
    {
        cgltf_mesh* cmesh = &(data->meshes[m]);
        if (!cmesh)
        {
            continue;
        }
        std::vector<Matrix44> positionMatrices;
        if (gltfMeshMatrixList.find(cmesh) != gltfMeshMatrixList.end())
        {
            positionMatrices = gltfMeshMatrixList[cmesh];
        }
        if (positionMatrices.empty())
        {
            positionMatrices.push_back(Matrix44::IDENTITY);
        }

        StringVec paths;
        if (gltfMeshPathList.find(cmesh) != gltfMeshPathList.end())
        {
            paths = gltfMeshPathList[cmesh];
        }
        if (paths.empty())
        {
            string meshName = cmesh->name ? string(cmesh->name) : DEFAULT_MESH_PREFIX + std::to_string(meshCount++);
            paths.push_back(meshName);
        }

        // Iterate through all parent transform
        for (size_t mtx = 0; mtx < positionMatrices.size(); mtx++)
        {
            const Matrix44& positionMatrix = positionMatrices[mtx];
            const Matrix44 normalMatrix = positionMatrix.getInverse().getTranspose();

            for (cgltf_size primitiveIndex = 0; primitiveIndex < cmesh->primitives_count; ++primitiveIndex)
            {
                cgltf_primitive* primitive = &cmesh->primitives[primitiveIndex];
                if (!primitive)
                {
                    continue;
                }

                if (primitive->type != cgltf_primitive_type_triangles)
                {
                    if (_debugLevel > 0)
                    {
                        std::cout << "Skip non-triangle indexed mesh: " << cmesh->name << std::endl;
                    }
                    continue;
                }

                Vector3 boxMin = { MAX_FLOAT, MAX_FLOAT, MAX_FLOAT };
                Vector3 boxMax = { -MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT };

                // Create a unique path for the mesh.
                string meshName = paths[mtx];
                while (meshNames.count(meshName))
                {
                    meshName = incrementName(meshName);
                }
                meshNames.insert(meshName);

                MeshPtr mesh = Mesh::create(meshName);
                if (_debugLevel > 0)
                {
                    std::cout << "Translate mesh: " << meshName << std::endl;
                }
                meshList.push_back(mesh);
                mesh->setSourceUri(filePath);

                MeshStreamPtr positionStream = nullptr;
                MeshStreamPtr normalStream = nullptr;
                MeshStreamPtr colorStream = nullptr;
                MeshStreamPtr texcoordStream = nullptr;
                MeshStreamPtr vec4TangentStream = nullptr;
                int colorAttrIndex = 0;

                // Read in vertex streams
                for (cgltf_size prim = 0; prim < primitive->attributes_count; prim++)
                {
                    cgltf_attribute* attribute = &primitive->attributes[prim];
                    cgltf_accessor* accessor = attribute->data;
                    if (!accessor)
                    {
                        continue;
                    }
                    // Only load one stream of each type for now.
                    cgltf_int streamIndex = attribute->index;
                    if (streamIndex != 0)
                    {
                        continue;
                    }

                    // Get data as floats
                    cgltf_size floatCount = cgltf_accessor_unpack_floats(accessor, NULL, 0);
                    std::vector<float> attributeData;
                    attributeData.resize(floatCount);
                    floatCount = cgltf_accessor_unpack_floats(accessor, &attributeData[0], floatCount);

                    cgltf_size vectorSize = cgltf_num_components(accessor->type);
                    size_t desiredVectorSize = 3;

                    MeshStreamPtr geomStream = nullptr;

                    bool isPositionStream = (attribute->type == cgltf_attribute_type_position);
                    bool isNormalStream = (attribute->type == cgltf_attribute_type_normal);
                    bool isTangentStream = (attribute->type == cgltf_attribute_type_tangent);
                    bool isColorStream = (attribute->type == cgltf_attribute_type_color);
                    bool isTexCoordStream = (attribute->type == cgltf_attribute_type_texcoord);

                    if (isPositionStream)
                    {
                        // Create position stream
                        positionStream = MeshStream::create("i_" + MeshStream::POSITION_ATTRIBUTE, MeshStream::POSITION_ATTRIBUTE, streamIndex);
                        mesh->addStream(positionStream);
                        geomStream = positionStream;
                    }
                    else if (isNormalStream)
                    {
                        normalStream = MeshStream::create("i_" + MeshStream::NORMAL_ATTRIBUTE, MeshStream::NORMAL_ATTRIBUTE, streamIndex);
                        mesh->addStream(normalStream);
                        geomStream = normalStream;
                    }
                    else if (isTangentStream)
                    {
                        vec4TangentStream = MeshStream::create("i_" + MeshStream::TANGENT_ATTRIBUTE + "4", MeshStream::TANGENT_ATTRIBUTE, streamIndex);
                        vec4TangentStream->setStride(MeshStream::STRIDE_4D); // glTF stores the bitangent sign in the 4th component
                        geomStream = vec4TangentStream;
                        desiredVectorSize = 4;
                    }
                    else if (isColorStream)
                    {
                        colorStream = MeshStream::create("i_" + MeshStream::COLOR_ATTRIBUTE + "_" + std::to_string(colorAttrIndex), MeshStream::COLOR_ATTRIBUTE, streamIndex);
                        mesh->addStream(colorStream);
                        geomStream = colorStream;
                        if (vectorSize == 4)
                        {
                            colorStream->setStride(MeshStream::STRIDE_4D);
                            desiredVectorSize = 4;
                        }
                        colorAttrIndex++;
                    }
                    else if (isTexCoordStream)
                    {
                        texcoordStream = MeshStream::create("i_" + MeshStream::TEXCOORD_ATTRIBUTE + "_0", MeshStream::TEXCOORD_ATTRIBUTE, 0);
                        mesh->addStream(texcoordStream);
                        if (vectorSize == 2)
                        {
                            texcoordStream->setStride(MeshStream::STRIDE_2D);
                            desiredVectorSize = 2;
                        }
                        geomStream = texcoordStream;
                    }
                    else
                    {
                        if (_debugLevel > 0)
                            std::cout << "Unknown stream type: " << std::to_string(attribute->type) << std::endl;
                    }

                    // Fill in stream
                    if (geomStream)
                    {
                        MeshFloatBuffer& buffer = geomStream->getData();
                        cgltf_size vertexCount = accessor->count;
                        geomStream->reserve(vertexCount);

                        if (_debugLevel > 0)
                        {
                            std::cout << "** Read stream: " << geomStream->getName() << std::endl;
                            std::cout << " - vertex count: " << std::to_string(vertexCount) << std::endl;
                            std::cout << " - vector size: " << std::to_string(vectorSize) << std::endl;
                        }

                        for (cgltf_size i = 0; i < vertexCount; i++)
                        {
                            const float* input = &attributeData[vectorSize * i];
                            if (isPositionStream)
                            {
                                Vector3 position;
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    // Update bounding box
                                    float floatValue = (v < vectorSize) ? input[v] : 0.0f;
                                    position[v] = floatValue;
                                }
                                position = positionMatrix.transformPoint(position);
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    buffer.push_back(position[v]);
                                    boxMin[v] = std::min(position[v], boxMin[v]);
                                    boxMax[v] = std::max(position[v], boxMax[v]);
                                }
                            }
                            else if (isNormalStream)
                            {
                                Vector3 normal;
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    float floatValue = (v < vectorSize) ? input[v] : 0.0f;
                                    normal[v] = floatValue;
                                }
                                normal = normalMatrix.transformVector(normal).getNormalized();
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    buffer.push_back(normal[v]);
                                }
                            }
                            else
                            {
                                for (cgltf_size v = 0; v < desiredVectorSize; v++)
                                {
                                    float floatValue = (v < vectorSize) ? input[v] : 0.0f;
                                    // Perform v-flip
                                    if (isTexCoordStream && v == 1)
                                    {
                                        if (!texcoordVerticalFlip)
                                        {
                                            floatValue = 1.0f - floatValue;
                                        }
                                    }
                                    buffer.push_back(floatValue);
                                }
                            }
                        }
                    }
                }

                if (!positionStream)
                {
                    continue;
                }

                // Read indexing
                MeshPartitionPtr part = MeshPartition::create();
                size_t indexCount = 0;
                cgltf_accessor* indexAccessor = primitive->indices;
                if (indexAccessor)
                {
                    indexCount = indexAccessor->count;
                }
                else
                {
                    indexCount = positionStream->getData().size();
                }
                size_t faceCount = indexCount / FACE_VERTEX_COUNT;
                part->setFaceCount(faceCount);
                part->setName(meshName);

                MeshIndexBuffer& indices = part->getIndices();
                if (_debugLevel > 0)
                {
                    std::cout << "** Read indexing: Count = " << std::to_string(indexCount) << std::endl;
                }
                if (indexAccessor)
                {
                    for (cgltf_size i = 0; i < indexCount; i++)
                    {
                        uint32_t vertexIndex = static_cast<uint32_t>(cgltf_accessor_read_index(indexAccessor, i));
                        indices.push_back(vertexIndex);
                    }
                }
                else
                {
                    for (cgltf_size i = 0; i < indexCount; i++)
                    {
                        indices.push_back(static_cast<uint32_t>(i));
                    }
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
            }
        }
    }

    return true;
}

MATERIALX_NAMESPACE_END

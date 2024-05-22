//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_TINYUSDZLOADER_H
#define MATERIALX_TINYUSDZLOADER_H

/// @file
/// USD format loader using the TinyUSDZ library

#include <MaterialXRender/GeometryHandler.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to a GLTFLoader
using TinyUSDZLoaderPtr = std::shared_ptr<class TinyUSDZLoader>;

/// @class TinyUSDZLoader
/// Wrapper for loader to read in USD files using the TinyUSDZ library.
class MX_RENDER_API TinyUSDZLoader : public GeometryLoader
{
  public:
    TinyUSDZLoader() :
        _debugLevel(0)
    {
        _extensions = { "usd", "USD", "usda", "USDA", "usdc", "USDC", "usdz", "USDZ" };
    }
    virtual ~TinyUSDZLoader() { }

    /// Create a new loader
    static TinyUSDZLoaderPtr create() { return std::make_shared<TinyUSDZLoader>(); }

    /// Load geometry from file path
    bool load(const FilePath& filePath, MeshList& meshList, bool texcoordVerticalFlip = false) override;

  private:
    unsigned int _debugLevel;
};

MATERIALX_NAMESPACE_END

#endif

rmdir /s /q build
mkdir build

cmake.exe -G "Visual Studio 17 2022" -A x64 -Bbuild -S. ^
  -DMATERIALX_BUILD_MONOLITHIC=On ^
  -DMATERIALX_BUILD_VIEWER=On

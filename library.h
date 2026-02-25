#pragma once

#ifdef _WIN32
    #define LIB_EXPORT __declspec(dllexport)
#else
    #define LIB_EXPORT
#endif

extern "C" {
    LIB_EXPORT void processImage(const char* imagePath, const char* modelPath, const char* outputPath);
}
#pragma once

#ifdef _WIN32
    #define LIB_EXPORT __declspec(dllexport)
#else
    #define LIB_EXPORT
#endif

extern "C" {
    LIB_EXPORT void processImage(const unsigned char* imageBuffer, int width, int height, int channels, const char* modelPath, const char* outputPath);
}
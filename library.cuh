#ifndef DEVELOPMENT_LIBRARY_CUH
#define DEVELOPMENT_LIBRARY_CUH

#ifdef _WIN32
    #define CUDA_LIB_EXPORT __declspec(dllexport)
#else
    #define CUDA_LIB_EXPORT __attribute__((visibility("default"))) // Лучше для Linux
#endif

// Это интерфейс для Java
extern "C" {
    // Передаем пути к картинке, модели и куда сохранить результат
    CUDA_LIB_EXPORT void processImage(const char* imagePath, const char* modelPath, const char* outputPath);
}

#endif //DEVELOPMENT_LIBRARY_CUH
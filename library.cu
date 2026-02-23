#include "library.cuh"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;

// This block give access to GPU kernel
__global__ void Kernel(uchar* data, int width, int height, size_t step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar* row = data + y * step;
        row[x] = row[x];
    }
}

// This block its specific "Main" and link for .so file
extern "C" {
    CUDA_LIB_EXPORT void processImage(const char* imagePath, const char* modelPath, const char* outputPath) {

        // Download image
        const Mat img = imread(imagePath, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Error: Could not read image from " << imagePath << endl;
            return;
        }

        // Use DNN and make calculation on GPU
        dnn_superres::DnnSuperResImpl res_impl;
        res_impl.readModel(modelPath);
        res_impl.setModel("lapsrn", 8);
        res_impl.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        res_impl.setPreferableTarget(dnn::DNN_TARGET_CUDA);

        cuda::GpuMat gpu_img;
        cuda::Stream stream;

        gpu_img.upload(img, stream);
        stream.waitForCompletion();


        Mat cpu_img, cpu_dst;
        gpu_img.download(cpu_img);
        res_impl.upsample(cpu_img, cpu_dst);

        // Return result after upscale to GPU, DNN dont have GPUmat ;(
        gpu_img.upload(cpu_dst, stream);
        cuda::GpuMat gpu_grey;

        // Convert to grayscale + threshold for letters + morphFilter
        cuda::cvtColor(gpu_img, gpu_grey, COLOR_BGR2GRAY, 0, stream);
        cuda::threshold(gpu_grey, gpu_grey, 115, 255, ADAPTIVE_THRESH_MEAN_C,  stream);

        Ptr<cuda::Filter> morphFilter = cuda::createMorphologyFilter(
            MORPH_CLOSE, gpu_grey.type(),
            getStructuringElement(MORPH_ELLIPSE, Size(3, 3))
        );
        morphFilter->apply(gpu_grey, gpu_grey, stream);

        // Use ALL performance our GPU
        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Kernel, 0, 0);

        int dimX = 32;
        int dimY = blockSize / dimX;
        if (blockSize < 32) {
            dimX = blockSize;
            dimY = 1;
        }

        dim3 threads(dimX, dimY);
        dim3 grid((gpu_grey.cols + threads.x - 1) / threads.x,
                  (gpu_grey.rows + threads.y - 1) / threads.y);

        // Call our Kernel
        Kernel<<<grid, threads, 0, static_cast<cudaStream_t>(stream.cudaPtr())>>>(
            gpu_grey.data,
            gpu_grey.cols,
            gpu_grey.rows,
            gpu_grey.step
        );

        stream.waitForCompletion();

        // Save image(debug)
        Mat result;
        gpu_grey.download(result);
        imwrite(outputPath, result);

        cout << "Processing complete. Saved to " << outputPath << endl;
    }
}
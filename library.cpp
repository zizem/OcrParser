#include "library.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/ximgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

extern "C" {
    // сигнатура processImage
    void processImage(const unsigned char* imageBuffer, int width, int height, int channels, const char* modelPath, const char* outputPath) {

        //  Загрузка зображения з буфера
        int cvType = CV_8UC3; //  3 канала
        if (channels == 1) cvType = CV_8UC1;
        else if (channels == 4) cvType = CV_8UC4;

        Mat img(height, width, cvType, (void*)imageBuffer);

        if (img.empty() || !img.data) {
            cerr << "Error: Could not read image from buffer" << endl;
            return;
        }

        Mat processedImage;
        //  приводим до 3-канального BGR формату
        if (channels == 4) {
            cvtColor(img, processedImage, COLOR_BGRA2BGR);
        } else if (channels == 1) {
            cvtColor(img, processedImage, COLOR_GRAY2BGR);
        } else {
            processedImage = img.clone();
        }


        // Апскейл
        if (modelPath != nullptr && string(modelPath) != "") {
            try {
                dnn_superres::DnnSuperResImpl res_impl;
                res_impl.readModel(modelPath);
                res_impl.setModel("lapsrn", 4); //  4x така модель
                res_impl.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
                res_impl.setPreferableTarget(dnn::DNN_TARGET_CPU);

                res_impl.upsample(processedImage, processedImage);
            } catch (const Exception& e) {
                cerr << "Warning: DNN SuperRes failed. " << e.what() << endl;
            }
        }

        //  Перевод в градацію сірого
        Mat gray;
        cvtColor(processedImage, gray, COLOR_BGR2GRAY);

        // Розмиття для видалення цифрового шума
        Mat denoised;
        GaussianBlur(gray, denoised, Size(3, 3), 0);

        // Бінаризація Sauvola
        Mat binary;
        /*
         * blockSize (51) - розмір локального вікна.
         * k (0.1)        - чувствительность (обычно от 0.1 до 0.2)
         */
        niBlackThreshold(denoised, binary, 255, THRESH_BINARY, 51, 0.1, BINARIZATION_SAUVOLA);
        if (mean(binary)[0] < 128) {
            bitwise_not(binary, binary);
        }

        // Очистка  маленьких точек
        // Локальная бинаризация .
        Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2)); // было 2 2
        morphologyEx(binary, binary, MORPH_OPEN, kernel);

        string tmpPath = (std::filesystem::temp_directory_path() / "ocr_tmp.png").string();
        //  Сохранение у тимчасову директорію
        imwrite(tmpPath, binary);
        std::ofstream flag(tmpPath + ".ready");

    }
}
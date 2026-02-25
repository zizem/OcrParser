#include "library.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;

// копирование пикселей на CPU
void processPixels(Mat& data) {
    for (int y = 0; y < data.rows; y++) {
        uchar* row = data.ptr<uchar>(y);
        for (int x = 0; x < data.cols; x++) {
            row[x] = row[x];
        }
    }
}

extern "C" {
    void processImage(const char* imagePath, const char* modelPath, const char* outputPath) {
        // Загрузка изображения
        const Mat img = imread(imagePath, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Error: Could not read image from " << imagePath << endl;
            return;
        }

        // Апскейл через DNN
        dnn_superres::DnnSuperResImpl res_impl;
        res_impl.readModel(modelPath);
        res_impl.setModel("lapsrn", 8);
        res_impl.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        res_impl.setPreferableTarget(dnn::DNN_TARGET_CPU);

        Mat upscaled;
        res_impl.upsample(img, upscaled);

        // Перевод и серый
        Mat grey;
        cvtColor(upscaled, grey, COLOR_BGR2GRAY);

        // Пороговая обработка
        threshold(grey, grey, 115, 255, THRESH_BINARY);

        // Морфологический фильтр
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(grey, grey, MORPH_CLOSE, kernel);


        processPixels(grey);

        imwrite(outputPath, grey);
        cout << "Processing complete. Saved to " << outputPath << endl;
    }
}
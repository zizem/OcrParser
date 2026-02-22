package org.example;

import java.io.IOException;

public class Main {
    static void main(String[] args) throws IOException {
        String imagePath = "/home/zizen/Cource/JavaFinal/src/main/java/org/example/test/NewSeria.png";
        String modelPath = "/home/zizen/Cource/JavaFinal/src/main/java/org/example/models/LapSRN_x8.pb";
        String outputPath = "/home/zizen/Cource/JavaFinal/src/main/java/org/example/output.png";

        System.out.println("Starting image processing...");
        try {
            LinkCudaSO.CudaWrapperLib.INSTANCES_LINUX.processImage(imagePath, modelPath, outputPath);

        } catch (Exception e) {
            System.out.println("Error processing image" + e.getMessage());
            e.printStackTrace();
        }

    }

}

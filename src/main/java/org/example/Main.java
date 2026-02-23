package org.example;

import org.example.Screen.ScreenSelector;

import javax.swing.*;
import java.io.IOException;

public class Main {
    static void main(String[] args) throws IOException {
        String imagePath = "/home/zizen/Cource/JavaFinal/src/main/java/NewSeria.png";
        String modelPath = "/home/zizen/Cource/JavaFinal/models/LapSRN_x8.pb";
        String outputPath = "/home/zizen/Cource/JavaFinal/src/main/java/org/example/output.png";


        SwingUtilities.invokeLater(() -> new ScreenSelector().setVisible(true));
        System.out.println("Starting image processing...");


    }

}

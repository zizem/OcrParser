package org.example.link;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.io.File;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;

public class NativeLib {
        //метод для підгрузки С++ модуля та сігнатурами
    public interface CourceLib extends Library {
        void processImage(byte[] imageBuffer, int width, int height, int channels, String modelPath, String outputPath);
    }

    public static final CourceLib INSTANCE;

    static {
        //перевірка яка система та пошук по директорії resources
        String os = System.getProperty("os.name").toLowerCase();
        String libFolder = os.contains("win") ? "win32-x86-64" : "linux-x86-64";
        String libName = os.contains("win") ? "cource.dll" : "cource.so";
        //шлях до ресурсів
        URL url = NativeLib.class.getResource("/" + libFolder + "/" + libName);

        if (url != null) {
            String rawPath = url.getPath();
            if (rawPath.startsWith("file:/")) rawPath = rawPath.substring(6);
            String dirPath = new File(URLDecoder.decode(rawPath, StandardCharsets.UTF_8)).getParent();

            System.out.println("Loading libraries from: " + dirPath);
            System.setProperty("jna.library.path", dirPath);

            if (os.contains("win")) {
                try {
                    // 1. Рантайм MinGW
                    loadLib(dirPath, "libwinpthread-1.dll");
                    loadLib(dirPath, "libgcc_s_seh-1.dll");
                    loadLib(dirPath, "libstdc++-6.dll");

                    // 2. Базовые модули OpenCV
                    loadLib(dirPath, "libopencv_core4120.dll");
                    loadLib(dirPath, "libopencv_flann4120.dll");
                    loadLib(dirPath, "libopencv_imgproc4120.dll");
                    loadLib(dirPath, "libopencv_imgcodecs4120.dll");
                    loadLib(dirPath, "libopencv_videoio4120.dll");
                    loadLib(dirPath, "opencv_videoio_ffmpeg4120_64.dll");

                    // 3. Средний уровень
                    loadLib(dirPath, "libopencv_features2d4120.dll");
                    loadLib(dirPath, "libopencv_calib3d4120.dll");
                    loadLib(dirPath, "libopencv_ml4120.dll");
                    loadLib(dirPath, "libopencv_dnn4120.dll");
                    loadLib(dirPath, "libopencv_photo4120.dll");
                    loadLib(dirPath, "libopencv_video4120.dll");
                    loadLib(dirPath, "libopencv_objdetect4120.dll");

                    // 4. Contrib модули
                    loadLib(dirPath, "libopencv_img_hash4120.dll");
                    loadLib(dirPath, "libopencv_plot4120.dll");
                    loadLib(dirPath, "libopencv_phase_unwrapping4120.dll");
                    loadLib(dirPath, "libopencv_aruco4120.dll");
                    loadLib(dirPath, "libopencv_bgsegm4120.dll");
                    loadLib(dirPath, "libopencv_bioinspired4120.dll");
                    loadLib(dirPath, "libopencv_dnn_superres4120.dll");
                    loadLib(dirPath, "libopencv_face4120.dll");
                    loadLib(dirPath, "libopencv_structured_light4120.dll");
                    loadLib(dirPath, "libopencv_text4120.dll");
                    loadLib(dirPath, "libopencv_tracking4120.dll");
                    loadLib(dirPath, "libopencv_xfeatures2d4120.dll");
                    loadLib(dirPath, "libopencv_ximgproc4120.dll");
                    loadLib(dirPath, "libopencv_xphoto4120.dll");
                    loadLib(dirPath, "libopencv_wechat_qrcode4120.dll");

                    // 5. OpenCV Java JNI
                    loadLib(dirPath, "libopencv_java4120.dll");

                    System.out.println("All dependencies loaded successfully.");
                } catch (UnsatisfiedLinkError e) {
                    System.err.println("Failed to pre-load dependencies: " + e.getMessage());
                }
            }
        } else {
            System.err.println("Library not found in classpath: " + libFolder + "/" + libName);
        }

        INSTANCE = Native.load("cource", CourceLib.class);
    }

    private static void loadLib(String path, String name) {
        File file = new File(path, name);
        if (file.exists()) {
            System.load(file.getAbsolutePath());
            System.out.println("Loaded: " + name);
        } else {
            System.err.println("Warning: not found: " + name);
        }
    }
}
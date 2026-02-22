package org.example;

import com.sun.jna.Library;
import com.sun.jna.Native;

public class LinkCudaSO {
    public interface CudaWrapperLib extends Library {

        CudaWrapperLib INSTANCES_LINUX = Native.load("cuda_wrapper", CudaWrapperLib.class);



        void processImage(String imagePath, String modelPath, String outputPath);
    }


}


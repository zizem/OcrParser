cmake_minimum_required(VERSION 3.20)
project(cource)

set(CMAKE_CXX_STANDARD 20)

# ─── Статическая линковка рантайма MinGW ───────────────────────────────────
if(MINGW)
    add_link_options(
            "-static-libgcc"
            "-static-libstdc++"
            "-Wl,-Bstatic,--whole-archive"
            "-lpthread"
            "-Wl,-Bdynamic,--no-whole-archive"
    )
endif()

# ─── OpenCV ────────────────────────────────────────────────────────────────
set(OpenCV_DIR "D:/opencv/mybuild/install/x64/mingw/lib")
find_package(OpenCV REQUIRED)

# ─── Ваша DLL для JNI/JNA ──────────────────────────────────────────────────
add_library(cource SHARED library.cpp)


set_target_properties(cource PROPERTIES PREFIX "")

target_include_directories(cource PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cource PRIVATE ${OpenCV_LIBS})

# ─── Пути к бинарникам ─────────────────────────────────────────────────────
set(OPENCV_BIN_DIR  "D:/opencv/mybuild/install/x64/mingw/bin")
set(OPENCV_JAVA_DIR "D:/opencv/mybuild/install/java")
set(MINGW_BIN_DIR   "C:/Users/zizem/AppData/Local/Programs/CLion/bin/mingw/bin")

# ─── Список OpenCV DLL ─────────────────────────────────────────────────────
set(OPENCV_DLLS
        "${OPENCV_BIN_DIR}/libopencv_core4120.dll"
        "${OPENCV_BIN_DIR}/libopencv_dnn4120.dll"
        "${OPENCV_BIN_DIR}/libopencv_imgcodecs4120.dll"
        "${OPENCV_BIN_DIR}/libopencv_imgproc4120.dll"
        "${OPENCV_JAVA_DIR}/libopencv_java4120.dll"
)

# ─── Рантайм MinGW ─────────────────────────────────────────────────────────
set(MINGW_DLLS
        "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
        "${MINGW_BIN_DIR}/libstdc++-6.dll"
        "${MINGW_BIN_DIR}/libwinpthread-1.dll"
)

# ─── Копирование после сборки ──────────────────────────────────────────────
foreach(DLL IN LISTS OPENCV_DLLS MINGW_DLLS)
    add_custom_command(TARGET cource POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}"
            $<TARGET_FILE_DIR:cource>
            COMMENT "Copying ${DLL}"
    )
endforeach()

# ─── Вывод информации при конфигурации ─────────────────────────────────────
message(STATUS "OpenCV version   : ${OpenCV_VERSION}")
message(STATUS "OpenCV libs      : ${OpenCV_LIBS}")
message(STATUS "OpenCV includes  : ${OpenCV_INCLUDE_DIRS}")
#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

// 설정 정보를 저장하는 구조체
struct AppConfig {
    struct {
        bool direct_conversion;
        
        struct {
            double clip_limit;
            int tile_grid_size;
        } clahe;
    } visualization;
    
    struct {
        struct {
            int width;
            int height;
            std::string format;
            int fps;
        } color, depth;
    } stream;
    
    struct {
        float min;
        float max;
    } depth_range;
    
    struct {
        std::string directory;
    } save;

    struct PoseConfig {
        std::string model_path; // .trt 모델 파일 경로
        bool use_cuda; // CUDA 사용 여부
        float confidence_threshold;
        int input_width;
        int input_height;
        int heatmap_width;
        int heatmap_height;
        std::vector<float> mean; // [R, G, B] 순서
        std::vector<float> std;  // [R, G, B] 순서
    } pose;
};

class ConfigManager {
public:
    static bool loadConfig(const std::string& config_file, AppConfig& config);
    static void setDefaultConfig(AppConfig& config);
    static void printConfig(const AppConfig& config);
}; 
#include "ConfigManager.h"
#include <iomanip>

bool ConfigManager::loadConfig(const std::string& config_file, AppConfig& config) {
    try {
        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "설정 파일을 열 수 없습니다: " << config_file << std::endl;
            return false;
        }
        
        fs["visualization"]["direct_conversion"] >> config.visualization.direct_conversion;
        fs["visualization"]["clahe"]["clip_limit"] >> config.visualization.clahe.clip_limit;
        fs["visualization"]["clahe"]["tile_grid_size"] >> config.visualization.clahe.tile_grid_size;

        fs["stream"]["color"]["width"] >> config.stream.color.width;
        fs["stream"]["color"]["height"] >> config.stream.color.height;
        fs["stream"]["color"]["format"] >> config.stream.color.format;
        fs["stream"]["color"]["fps"] >> config.stream.color.fps;
        
        fs["stream"]["depth"]["width"] >> config.stream.depth.width;
        fs["stream"]["depth"]["height"] >> config.stream.depth.height;
        fs["stream"]["depth"]["format"] >> config.stream.depth.format;
        fs["stream"]["depth"]["fps"] >> config.stream.depth.fps;

        fs["depth_range"]["min"] >> config.depth_range.min;
        fs["depth_range"]["max"] >> config.depth_range.max;

        fs["save"]["directory"] >> config.save.directory;
        
        fs.release();
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "설정 파일 로드 오류: " << e.what() << std::endl;
        return false;
    }
}

void ConfigManager::setDefaultConfig(AppConfig& config) {
    // 기본 설정값 설정
    config.visualization.direct_conversion = true;
    config.visualization.clahe.clip_limit = 3.0;
    config.visualization.clahe.tile_grid_size = 16;
    
    config.stream.color.width = 640;
    config.stream.color.height = 480;
    config.stream.color.format = "BGR8";
    config.stream.color.fps = 30;
    
    config.stream.depth.width = 640;
    config.stream.depth.height = 480;
    config.stream.depth.format = "Z16";
    config.stream.depth.fps = 30;
    
    config.depth_range.min = 0.1f;
    config.depth_range.max = 1.0f;
    
    config.save.directory = "./results/";
}

void ConfigManager::printConfig(const AppConfig& config) {
    std::cout << "====== 현재 설정 ======" << std::endl;
    
    std::cout << "[시각화 설정]" << std::endl;
    std::cout << "  - 변환 방식: " << (config.visualization.direct_conversion ? "직접 변환 (32비트->8비트)" : "단계별 변환 (32비트->16비트->8비트)") << std::endl;
    std::cout << "  - CLAHE 설정:" << std::endl;
    std::cout << "    * Clip Limit: " << config.visualization.clahe.clip_limit << std::endl;
    std::cout << "    * Tile Grid Size: " << config.visualization.clahe.tile_grid_size << "x" << config.visualization.clahe.tile_grid_size << std::endl;
    
    std::cout << "[스트림 설정]" << std::endl;
    std::cout << "  - 컬러 스트림: " << config.stream.color.width << "x" << config.stream.color.height 
              << ", 포맷: " << config.stream.color.format << ", FPS: " << config.stream.color.fps << std::endl;
    std::cout << "  - 깊이 스트림: " << config.stream.depth.width << "x" << config.stream.depth.height 
              << ", 포맷: " << config.stream.depth.format << ", FPS: " << config.stream.depth.fps << std::endl;
    
    std::cout << "[깊이 범위 설정]" << std::endl;
    std::cout << "  - 최소 깊이: " << config.depth_range.min << "m" << std::endl;
    std::cout << "  - 최대 깊이: " << config.depth_range.max << "m" << std::endl;
    
    std::cout << "[저장 설정]" << std::endl;
    std::cout << "  - 저장 디렉토리: " << config.save.directory << std::endl;
    
    std::cout << "======================" << std::endl;
} 
//g++ -o main main.cpp ConfigManager.cpp utils/FileUtils.cpp DepthProcessor.cpp utils/FPSCounter.cpp RealSenseCamera.cpp utils/ImageSaver.cpp utils/KeyboardHandler.cpp PoseEstimator.cpp -lrealsense2 -lcudart -lnvinfer `pkg-config --cflags --libs opencv4`
#include <iostream>
#include <string>
#include <unistd.h>
#include <limits.h>
#include "ConfigManager.h"
#include "utils/FileUtils.h"
#include "DepthProcessor.h"
#include "utils/FPSCounter.h"
#include "RealSenseCamera.h"
#include "utils/ImageSaver.h"
#include "utils/KeyboardHandler.h"
#include "PoseEstimator.h"
#include "utils/Visualizer.h"

// 소스 디렉토리 경로 얻기
std::string getSourceDirectory() {
    char currentDir[PATH_MAX];
    if (getcwd(currentDir, sizeof(currentDir)) != nullptr) {
        return std::string(currentDir);
    }
    return ".";
}

int main(int argc, char *argv[]) try
{
    // 소스 디렉토리의 config.yaml 직접 로드
    std::string sourceDir = getSourceDirectory();
    std::string configFile = sourceDir + "/config.yaml";
    
    std::cout << "설정 파일 경로: " << configFile << std::endl;
    
    // 설정 로드
    AppConfig config;
    if (!ConfigManager::loadConfig(configFile, config)) {
        std::cerr << "기본 설정을 사용합니다." << std::endl;
        ConfigManager::setDefaultConfig(config);
    }
    
    // 설정 정보 출력
    ConfigManager::printConfig(config);
    
    // TensorRT 포즈 추정 모델 로드 (Config에서 경로 사용)
    PoseEstimator poseEstimator(config);
    
    // 이미지 저장기 초기화
    Utils::ImageSaver imageSaver(config.save.directory);
    if (!imageSaver.prepareFolder()) {
        return EXIT_FAILURE;
    }
    
    // RealSense 카메라 초기화
    RealSenseCamera camera(config);
    if (!camera.start()) {
        std::cerr << "RealSense 카메라 시작 실패" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "RealSense 카메라 시작됨. 's'를 누르면 이미지와 깊이 맵을 저장하고, 'q'를 누르면 종료합니다." << std::endl;
    std::cout << "파일은 " << config.save.directory << "resultN/ 디렉토리에 저장됩니다." << std::endl;
    
    // FPS 카운터 초기화
    Utils::FPSCounter fpsCounter;
    
    // 키보드 핸들러 초기화
    Utils::KeyboardHandler keyboard;
    
    // 시각화 창 생성 - Visualizer 사용
    Utils::Visualizer::initializeWindows();
    
    std::cout << "'s'를 눌러서 저장하고, 'q'를 눌러서 종료하세요." << std::endl;
    
    // 메인 루프
    while(!keyboard.isQuitPressed()) {
        // FPS 업데이트
        float fps = fpsCounter.update();
        
        // 프레임 가져오기
        rs2::frameset frames;
        if(!camera.getFrames(frames)) {
            continue;
        }
        
        // 프레임셋에서 깊이 및 컬러 프레임 추출
        rs2::depth_frame depthFrame = frames.get_depth_frame();
        rs2::video_frame colorFrame = frames.get_color_frame();
        
        if (!depthFrame || !colorFrame) {
            std::cerr << "유효하지 않은 프레임 발견. 건너뜁니다." << std::endl;
            continue;
        }
        
        // 컬러 이미지를 OpenCV 형식으로 변환
        cv::Mat colorImage(cv::Size(colorFrame.get_width(), colorFrame.get_height()), 
                           CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);
        
        // 깊이 맵 시각화
        cv::Mat enhancedDepth = DepthProcessor::enhancedDepthVisualization(depthFrame, config);
        
        // 중앙 지점의 거리 정보 계산 - DepthProcessor 클래스 함수 사용
        float centerDist = DepthProcessor::calculateCenterDistance(depthFrame, config.depth_range.max);
        
        // 포즈 추정 실행
        std::vector<std::vector<cv::Point>> keypoints;
        bool success = poseEstimator.detect(colorImage, keypoints);
        
        // 포즈 추정 결과 시각화
        cv::Mat& poseImage = colorImage;
        
        // Visualizer를 사용하여 결과 그리기 및 표시
        Utils::Visualizer::drawResults(poseImage, enhancedDepth, keypoints, fps, centerDist, config);
        
        // 키 입력 대기 (1ms)
        keyboard.waitKey(1);
        
        // 's' 키를 누르면 이미지와 깊이 맵 저장
        if (keyboard.isSavePressed()) {
            // 포즈 추정 결과도 함께 저장
            imageSaver.saveImages(poseImage, enhancedDepth, depthFrame);
        }
    }
    
    Utils::Visualizer::destroyWindows();
    
    return EXIT_SUCCESS; 
}
catch (const rs2::error & e) 
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE; 
}
catch (const std::exception& e) 
{
    std::cerr << e.what() << std::endl; 
    return EXIT_FAILURE;  
}
#pragma once

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "ConfigManager.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

class DepthProcessor {
public:
    // 깊이 이미지 시각화 함수 - 설정에 따라 변환 방식 선택
    static cv::Mat enhancedDepthVisualization(const rs2::depth_frame& depthFrame, const AppConfig& config);
    
    // 깊이 맵을 바이너리 파일로 저장하는 함수
    static void saveDepthToBin(const rs2::depth_frame& depthFrame, const std::string& filename);
    
    // 중앙 지점의 거리 계산 함수 (픽셀 평균으로 안정성 향상)
    static float calculateCenterDistance(const rs2::depth_frame& depthFrame, float maxDepth, int windowSize = 5);
    
    // 이미지에 십자선 그리는 함수
    static void drawCrosshair(cv::Mat& image, int size = 10, const cv::Scalar& color = cv::Scalar(255, 255, 255));
    
private:
    // 직접 변환 방식 (32비트 -> 8비트)
    static cv::Mat directConversion(const rs2::depth_frame& depthFrame, float minDepth, float maxDepth);
    
    // 단계별 변환 방식 (32비트 -> 16비트 -> 8비트)
    static cv::Mat stepByStepConversion(const rs2::depth_frame& depthFrame, float minDepth, float maxDepth);
    
    // CLAHE 적용
    static cv::Mat applyCLAHE(const cv::Mat& depthImage, double clipLimit, int tileGridSize);
}; 
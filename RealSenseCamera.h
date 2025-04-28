#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "ConfigManager.h"

class RealSenseCamera {
public:
    RealSenseCamera(const AppConfig& config);
    ~RealSenseCamera();
    
    // 카메라 시작
    bool start();
    
    // 프레임 가져오기
    bool getFrames(rs2::frameset& frames);
    
private:
    rs2::pipeline pipe;
    rs2::config cfg;
    const AppConfig& config;
    
    // 포맷 문자열을 rs2_format으로 변환
    rs2_format getColorFormat(const std::string& format);
    rs2_format getDepthFormat(const std::string& format);
}; 
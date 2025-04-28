#include "RealSenseCamera.h"
#include <iostream>

RealSenseCamera::RealSenseCamera(const AppConfig& config) : config(config) {
}

RealSenseCamera::~RealSenseCamera() {
    // 파이프라인이 시작된 경우, 정리
    try {
        pipe.stop();
    } catch (const rs2::error& e) {
        std::cerr << "RealSense 에러 (파이프라인 정지 중): " << e.what() << std::endl;
    }
}

bool RealSenseCamera::start() {
    try {
        // 설정 파일에서 스트림 포맷 변환
        rs2_format colorFormat = getColorFormat(config.stream.color.format);
        rs2_format depthFormat = getDepthFormat(config.stream.depth.format);

        cfg.enable_stream(RS2_STREAM_COLOR, 
                        config.stream.color.width, 
                        config.stream.color.height, 
                        colorFormat, 
                        config.stream.color.fps);
        
        cfg.enable_stream(RS2_STREAM_DEPTH, 
                        config.stream.depth.width, 
                        config.stream.depth.height, 
                        depthFormat, 
                        config.stream.depth.fps);
        
        // 파이프라인 시작
        pipe.start(cfg);
        return true;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense 에러 (파이프라인 시작 중): " << e.what() << std::endl;
        return false;
    }
}

bool RealSenseCamera::getFrames(rs2::frameset& frames) {
    try {
        frames = pipe.wait_for_frames();
        return true;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense 에러 (프레임 대기 중): " << e.what() << std::endl;
        return false;
    }
}

rs2_format RealSenseCamera::getColorFormat(const std::string& format) {
    if (format == "RGB8") return RS2_FORMAT_RGB8;
    else if (format == "RGBA8") return RS2_FORMAT_RGBA8;
    else if (format == "BGRA8") return RS2_FORMAT_BGRA8;
    else return RS2_FORMAT_BGR8; // 기본값
}

rs2_format RealSenseCamera::getDepthFormat(const std::string& format) {
    if (format == "Z16") return RS2_FORMAT_Z16;
    else return RS2_FORMAT_Z16; // 현재는 Z16만 지원
} 
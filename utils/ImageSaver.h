#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "FileUtils.h"
#include "../DepthProcessor.h"

namespace Utils {
    class ImageSaver {
    public:
        ImageSaver(const std::string& baseDirectory);
        
        // 이미지 저장 폴더 준비
        bool prepareFolder();
        
        // 이미지 저장
        bool saveImages(const cv::Mat& colorImage, const cv::Mat& depthColormap, const rs2::depth_frame& depthFrame);
        
        // 현재 폴더 번호 반환
        int getCurrentFolderNumber() const;
        
    private:
        std::string baseDirectory;
        int folderNumber;
    };
} 
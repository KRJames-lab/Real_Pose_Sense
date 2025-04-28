#include "ImageSaver.h"
#include <iostream>
#include <sstream>

namespace Utils {
    ImageSaver::ImageSaver(const std::string& baseDirectory) : baseDirectory(baseDirectory) {
        // 시작 폴더 번호 찾기
        folderNumber = FileUtils::findNextResultFolder(baseDirectory);
    }

    bool ImageSaver::prepareFolder() {
        // 베이스 디렉토리 생성
        if (!FileUtils::createDirectory(baseDirectory)) {
            std::cerr << "결과 디렉토리 생성 실패: " << baseDirectory << std::endl;
            return false;
        }
        
        return true;
    }

    bool ImageSaver::saveImages(const cv::Mat& colorImage, const cv::Mat& depthColormap, const rs2::depth_frame& depthFrame) {
        // 폴더 경로 생성
        std::stringstream folderSs;
        folderSs << baseDirectory << "result" << folderNumber << "/";
        std::string resultFolder = folderSs.str();
        
        // 폴더 생성
        if (!FileUtils::createDirectory(resultFolder)) {
            std::cerr << "폴더 생성 실패: " << resultFolder << std::endl;
            return false;
        }
        
        // 파일 경로 생성
        std::string colorFilename = resultFolder + "color.png";
        std::string depthColormapFilename = resultFolder + "depth_colormap.png";
        std::string depthBinFilename = resultFolder + "depth.bin";
        
        // 이미지 저장
        cv::imwrite(colorFilename, colorImage);
        cv::imwrite(depthColormapFilename, depthColormap);
        DepthProcessor::saveDepthToBin(depthFrame, depthBinFilename);
        
        std::cout << "파일이 저장되었습니다. 폴더: result" << folderNumber << std::endl;
        
        // 폴더 번호 증가
        folderNumber++;
        
        return true;
    }

    int ImageSaver::getCurrentFolderNumber() const {
        return folderNumber;
    }
} 
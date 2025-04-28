#include "DepthProcessor.h"

cv::Mat DepthProcessor::enhancedDepthVisualization(const rs2::depth_frame& depthFrame, const AppConfig& config) {
    float minDepth = config.depth_range.min;
    float maxDepth = config.depth_range.max;
    bool directConversion = config.visualization.direct_conversion;
    
    cv::Mat depth8bit;
    
    if (directConversion) {
        depth8bit = DepthProcessor::directConversion(depthFrame, minDepth, maxDepth);
    } else {
        depth8bit = DepthProcessor::stepByStepConversion(depthFrame, minDepth, maxDepth);
    }
    
    // CLAHE 적용
    cv::Mat enhancedDepth = DepthProcessor::applyCLAHE(depth8bit, 
                                                     config.visualization.clahe.clip_limit, 
                                                     config.visualization.clahe.tile_grid_size);
    
    // 히트맵으로 변환
    cv::Mat colormap;
    cv::applyColorMap(enhancedDepth, colormap, cv::COLORMAP_TURBO);
    
    // 범위 정보를 이미지에 추가
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << minDepth << "m ~ " << maxDepth << "m";
    cv::putText(colormap, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // 중앙 지점의 거리 정보 계산 (calculateCenterDistance 함수 사용)
    float centerDist = calculateCenterDistance(depthFrame, maxDepth);
    
    // 중앙에 십자선 그리기
    drawCrosshair(colormap, 10, cv::Scalar(255, 255, 255));
    
    // 중앙 거리 정보 표시
    std::stringstream centerSs;
    centerSs << std::fixed << std::setprecision(2) << "Distance: " << centerDist << "m";
    cv::putText(colormap, centerSs.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    return colormap;
}

// 십자선 그리는 함수 구현
void DepthProcessor::drawCrosshair(cv::Mat& image, int size, const cv::Scalar& color) {
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    cv::line(image, cv::Point(centerX - size, centerY), cv::Point(centerX + size, centerY), color, 2);
    cv::line(image, cv::Point(centerX, centerY - size), cv::Point(centerX, centerY + size), color, 2);
}

// 중앙 지점의 거리 계산 함수 구현
float DepthProcessor::calculateCenterDistance(const rs2::depth_frame& depthFrame, float maxDepth, int windowSize) {
    int centerX = depthFrame.get_width() / 2;
    int centerY = depthFrame.get_height() / 2;
    float centerDist = 0.0f;
    int validCount = 0;
    
    // 중앙 주변 픽셀의 평균 거리 계산
    for (int y = centerY - windowSize/2; y <= centerY + windowSize/2; y++) {
        for (int x = centerX - windowSize/2; x <= centerX + windowSize/2; x++) {
            if (y >= 0 && y < depthFrame.get_height() && x >= 0 && x < depthFrame.get_width()) {
                float dist = depthFrame.get_distance(x, y);
                if (dist > 0.001f && dist < maxDepth) { // 유효한 깊이 값만 포함
                    centerDist += dist;
                    validCount++;
                }
            }
        }
    }
    
    // 유효한 값이 있으면 평균 계산
    if (validCount > 0) {
        centerDist /= validCount;
    } else {
        // 유효한 깊이 값이 없으면 단일 중앙 픽셀 시도
        centerDist = depthFrame.get_distance(centerX, centerY);
    }
    
    return centerDist;
}

cv::Mat DepthProcessor::directConversion(const rs2::depth_frame& depthFrame, float minDepth, float maxDepth) {
    // 깊이 프레임에서 데이터 가져오기
    int width = depthFrame.get_width();
    int height = depthFrame.get_height();
    
    // 32비트 float -> 8비트 변환 방식
    cv::Mat depthFloat(height, width, CV_32FC1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float depthValue = depthFrame.get_distance(x, y);
            
            if (depthValue < minDepth || depthValue > maxDepth || depthValue <= 0) {
                depthFloat.at<float>(y, x) = 0.0f;
            } else {
                depthFloat.at<float>(y, x) = depthValue;
            }
        }
    }
    
    // 깊이 값 정규화 (min_depth ~ max_depth -> 0.0 ~ 1.0)
    cv::Mat normalizedDepth;
    cv::subtract(depthFloat, minDepth, normalizedDepth);
    cv::divide(normalizedDepth, (maxDepth - minDepth), normalizedDepth);
    
    // 정규화된 float 이미지를 8비트로 직접 변환 (0.0~1.0 -> 0~255)
    cv::Mat depth8bit;
    normalizedDepth.convertTo(depth8bit, CV_8UC1, 255.0);
    
    return depth8bit;
}

cv::Mat DepthProcessor::stepByStepConversion(const rs2::depth_frame& depthFrame, float minDepth, float maxDepth) {
    // 깊이 프레임에서 데이터 가져오기
    int width = depthFrame.get_width();
    int height = depthFrame.get_height();
    
    // 32비트 float -> 16비트 -> 8비트 변환
    cv::Mat depthImage(height, width, CV_16UC1);
    
    // 깊이 데이터를 16비트(0-65535) 범위로 정규화
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float depthValue = depthFrame.get_distance(x, y);
            
            if (depthValue < minDepth || depthValue > maxDepth || depthValue <= 0) {
                depthImage.at<ushort>(y, x) = 0;
            } else {
                // 깊이값을 0-65535 범위로 변환 (16비트)
                depthImage.at<ushort>(y, x) = static_cast<ushort>((depthValue - minDepth) / (maxDepth - minDepth) * 65535);
            }
        }
    }
    
    // 16비트에서 8비트로 변환
    cv::Mat depth8bit;
    depthImage.convertTo(depth8bit, CV_8UC1, 1.0/256.0);
    
    return depth8bit;
}

cv::Mat DepthProcessor::applyCLAHE(const cv::Mat& depthImage, double clipLimit, int tileGridSize) {
    // CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->setTilesGridSize(cv::Size(tileGridSize, tileGridSize));
    
    cv::Mat enhancedDepth;
    clahe->apply(depthImage, enhancedDepth);
    
    return enhancedDepth;
}

void DepthProcessor::saveDepthToBin(const rs2::depth_frame& depthFrame, const std::string& filename) {
    // 깊이 프레임에서 데이터 가져오기
    int width = depthFrame.get_width();
    int height = depthFrame.get_height();
    
    // 바이너리 파일 열기
    std::ofstream outfile(filename, std::ios::binary);
    
    if (!outfile) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }
    
    // 파일 헤더 (너비와 높이 저장)
    outfile.write(reinterpret_cast<const char*>(&width), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&height), sizeof(int));
    
    // 모든 픽셀의 깊이 값을 저장
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float depthValue = depthFrame.get_distance(x, y);
            outfile.write(reinterpret_cast<const char*>(&depthValue), sizeof(float));
        }
    }
    
    outfile.close();
    std::cout << "깊이 맵이 저장되었습니다: " << filename << std::endl;
} 
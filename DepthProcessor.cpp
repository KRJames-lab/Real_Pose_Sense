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
    int width = depthFrame.get_width();
    int height = depthFrame.get_height();
    
    // 깊이 프레임 데이터를 float 타입의 Mat으로 복사 (get_distance 사용 유지)
    cv::Mat depthFloat(height, width, CV_32FC1);
    for (int y = 0; y < height; y++) {
        float* rowPtr = depthFloat.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            rowPtr[x] = depthFrame.get_distance(x, y);
        }
    }

    // 유효 범위를 벗어나는 픽셀 마스크 생성 (minDepth <= depth <= maxDepth, depth > 0)
    cv::Mat validMask;
    cv::inRange(depthFloat, cv::Scalar(minDepth), cv::Scalar(maxDepth), validMask);
    
    // 0 이하인 픽셀 마스크 생성
    cv::Mat positiveMask;
    cv::compare(depthFloat, 0.0f, positiveMask, cv::CMP_GT);

    // 두 마스크 결합 (유효 범위이면서 양수인 픽셀)
    validMask &= positiveMask;

    // 유효하지 않은 픽셀을 0으로 설정
    cv::Mat maskedDepth;
    depthFloat.copyTo(maskedDepth, validMask); // 유효한 값만 복사, 나머지는 0
    
    // 깊이 값 정규화 (min_depth ~ max_depth -> 0.0 ~ 1.0)
    cv::Mat normalizedDepth = cv::Mat::zeros(height, width, CV_32FC1); // 0으로 초기화
    float range = maxDepth - minDepth;
    
    // 0으로 나누는 경우 방지
    if (range > 1e-6) {
        cv::Mat temp = cv::Mat::zeros(height, width, CV_32FC1);
        // 유효 픽셀에 대해 (depth - minDepth) 계산
        cv::subtract(depthFloat, minDepth, temp, validMask);
        // 전체 temp 행렬에 대해 range로 나누기 (유효하지 않은 영역은 0/range = 0)
        cv::divide(temp, range, temp);
        // 마스크를 사용해 유효한 결과만 normalizedDepth에 복사
        temp.copyTo(normalizedDepth, validMask);
    }

    // 정규화된 float 이미지를 8비트로 변환 (0.0~1.0 -> 0~255)
    cv::Mat depth8bit;
    normalizedDepth.convertTo(depth8bit, CV_8UC1, 255.0);
    
    return depth8bit;
}

cv::Mat DepthProcessor::stepByStepConversion(const rs2::depth_frame& depthFrame, float minDepth, float maxDepth) {
    int width = depthFrame.get_width();
    int height = depthFrame.get_height();

    // 깊이 프레임 데이터를 float 타입의 Mat으로 복사 (get_distance 사용 유지)
    cv::Mat depthFloat(height, width, CV_32FC1);
     for (int y = 0; y < height; y++) {
        float* rowPtr = depthFloat.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            rowPtr[x] = depthFrame.get_distance(x, y);
        }
    }

    // 유효 범위를 벗어나는 픽셀 마스크 생성 (minDepth <= depth <= maxDepth, depth > 0)
    cv::Mat validMask;
    cv::inRange(depthFloat, cv::Scalar(minDepth), cv::Scalar(maxDepth), validMask);
    
    // 0 이하인 픽셀 마스크 생성
    cv::Mat positiveMask;
    cv::compare(depthFloat, 0.0f, positiveMask, cv::CMP_GT);

    // 두 마스크 결합 (유효 범위이면서 양수인 픽셀)
    validMask &= positiveMask;

    // 깊이 데이터를 16비트(0-65535) 범위로 정규화 (유효 픽셀 대상)
    cv::Mat depth16bit = cv::Mat::zeros(height, width, CV_16UC1); // 0으로 초기화
    float range = maxDepth - minDepth; // range 변수 추가

    // 0으로 나누는 경우 방지
    if (range > 1e-6) {
        float scale = 65535.0f / range;
        cv::Mat temp = cv::Mat::zeros(height, width, CV_32FC1);
        // 유효 픽셀에 대해 (depth - minDepth) 계산
        cv::subtract(depthFloat, minDepth, temp, validMask);
        // 전체 temp 행렬에 대해 스케일링 및 타입 변환
        cv::Mat temp16bit;
        temp.convertTo(temp16bit, CV_16UC1, scale);
        // 마스크를 사용해 유효한 결과만 depth16bit에 복사
        temp16bit.copyTo(depth16bit, validMask);
    }

    // 16비트에서 8비트로 변환
    cv::Mat depth8bit;
    depth16bit.convertTo(depth8bit, CV_8UC1, 1.0/256.0);
    
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
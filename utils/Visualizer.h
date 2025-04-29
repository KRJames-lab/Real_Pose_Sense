#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "../ConfigManager.h" // AppConfig 사용을 위해 포함 (상대 경로 또는 include path 설정 필요)
#include "../PoseEstimator.h" // drawKeypoints 사용 위해 포함
#include "../DepthProcessor.h" // drawCrosshair 사용 위해 포함

namespace Utils {
    namespace Visualizer {

        // 시각화 창 초기화
        void initializeWindows();

        // 결과 시각화 및 창 업데이트
        // - poseImage: 포즈 및 기타 정보가 그려질 대상 이미지 (수정됨)
        // - enhancedDepth: 시각화된 깊이 이미지
        // - keypoints: 검출된 키포인트
        // - fps: 현재 FPS 값
        // - centerDist: 중앙 거리 값
        // - config: 애플리케이션 설정 (필요시 사용)
        void drawResults(
            cv::Mat& poseImage, // 입력 이미지를 직접 수정
            const cv::Mat& enhancedDepth,
            const std::vector<std::vector<cv::Point>>& keypoints,
            float fps,
            float centerDist,
            const AppConfig& config // config는 const 참조로 받음
        );

        // 모든 시각화 창 닫기
        void destroyWindows();

    } // namespace Visualizer
} // namespace Utils 
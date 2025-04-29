#include "Visualizer.h"
#include <sstream> // stringstream 사용
#include <iomanip> // setprecision 사용

namespace Utils {
    namespace Visualizer {

        const std::string POSE_WINDOW_NAME = "Pose Estimation";
        const std::string DEPTH_WINDOW_NAME = "Enhanced Depth";

        void initializeWindows() {
            cv::namedWindow(POSE_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
            cv::namedWindow(DEPTH_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
        }

        void drawResults(
            cv::Mat& poseImage, // 입력 이미지를 직접 수정 (colorImage.clone() 대신 원본 사용 가정)
            const cv::Mat& enhancedDepth,
            const std::vector<std::vector<cv::Point>>& keypoints,
            float fps,
            float centerDist,
            const AppConfig& config // config는 현재 직접 사용되지 않지만, 향후 확장을 위해 남겨둠
        ) {
            // 포즈 추정 결과 시각화 (PoseEstimator의 static 함수 호출)
            if (!keypoints.empty()) {
                PoseEstimator::drawKeypoints(poseImage, keypoints);
            }
            
            // 중앙에 십자선 그리기 (DepthProcessor의 static 함수 호출)
            DepthProcessor::drawCrosshair(poseImage, 5, cv::Scalar(0, 255, 0));
            
            // FPS 정보 추가
            std::stringstream fpsSs;
            fpsSs << "FPS: " << std::fixed << std::setprecision(1) << fps;
            cv::putText(poseImage, fpsSs.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            // 거리 정보 추가
            std::stringstream distSs;
            distSs << "Distance: " << std::fixed << std::setprecision(2) << centerDist << "m";
            cv::putText(poseImage, distSs.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            // 컨트롤 정보 추가 - Pose Estimation에만 표시
            cv::putText(poseImage, "s: Save, q: Quit", cv::Point(10, poseImage.rows - 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
            
            // Pose Estimation과 Enhanced Depth 창 표시
            cv::imshow(POSE_WINDOW_NAME, poseImage);
            cv::imshow(DEPTH_WINDOW_NAME, enhancedDepth);
        }

        void destroyWindows() {
            cv::destroyAllWindows();
        }

    } // namespace Visualizer
} // namespace Utils 
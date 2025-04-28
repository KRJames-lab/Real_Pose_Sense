#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include <string>
#include <memory>

// Logger 클래스 정의 (NvInfer의 ILogger 구현)
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// TensorRT 엔진 소멸자
struct TRTDestroy {
    template <class T>
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

// 포즈 추정 클래스
class PoseEstimator {
public:
    // 생성자: TensorRT 모델 경로를 받아 초기화
    PoseEstimator(const std::string& modelPath);
    ~PoseEstimator();

    // 이미지에서 포즈 추정 실행
    bool detect(const cv::Mat& image, std::vector<std::vector<cv::Point>>& keypoints);
    
    // 이미지에 키포인트 그리기
    static void drawKeypoints(cv::Mat& image, const std::vector<std::vector<cv::Point>>& keypoints);

private:
    // TensorRT 관련 변수
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context;
    
    // 모델 관련 변수
    int inputH;
    int inputW;
    int outputSize;
    int numKeypoints;
    int batchSize;
    
    // 입출력 버퍼 - 최대 바인딩 개수를 3으로 변경 (1 입력 + 2 출력)
    void* buffers[3];
    int inputIndex;
    int outputIndex;
    
    // 전처리 함수: OpenCV Mat을 TensorRT 입력 형식으로 변환
    void preprocess(const cv::Mat& image, float* inputBuffer);
    
    // 후처리 함수: TensorRT 출력을 키포인트 목록으로 변환
    void postprocess(float* outputBuffer, const cv::Size& originalSize, std::vector<std::vector<cv::Point>>& keypoints);
    
    // TensorRT 엔진 로드
    bool loadEngine(const std::string& enginePath);
}; 
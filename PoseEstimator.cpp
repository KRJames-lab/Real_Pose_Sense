#include "PoseEstimator.h"
#include <fstream>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv2/dnn.hpp>

// Logger 구현
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "TensorRT: " << msg << std::endl;
    }
}

// COCO 키포인트 색상
const cv::Scalar colors[] = {
    cv::Scalar(255, 0, 0),     // 코
    cv::Scalar(255, 85, 0),    // 왼쪽 눈
    cv::Scalar(255, 170, 0),   // 오른쪽 눈
    cv::Scalar(255, 255, 0),   // 왼쪽 귀
    cv::Scalar(170, 255, 0),   // 오른쪽 귀
    cv::Scalar(85, 255, 0),    // 왼쪽 어깨
    cv::Scalar(0, 255, 0),     // 오른쪽 어깨
    cv::Scalar(0, 255, 85),    // 왼쪽 팔꿈치
    cv::Scalar(0, 255, 170),   // 오른쪽 팔꿈치
    cv::Scalar(0, 255, 255),   // 왼쪽 손목
    cv::Scalar(0, 170, 255),   // 오른쪽 손목
    cv::Scalar(0, 85, 255),    // 왼쪽 엉덩이
    cv::Scalar(0, 0, 255),     // 오른쪽 엉덩이
    cv::Scalar(85, 0, 255),    // 왼쪽 무릎
    cv::Scalar(170, 0, 255),   // 오른쪽 무릎
    cv::Scalar(255, 0, 255),   // 왼쪽 발목
    cv::Scalar(255, 0, 170)    // 오른쪽 발목
};

// COCO 키포인트 연결 정보
const std::vector<std::pair<int, int>> skeleton = {
    {5, 6},   // 어깨 - 어깨
    {5, 7},   // 왼쪽 어깨 - 왼쪽 팔꿈치
    {7, 9},   // 왼쪽 팔꿈치 - 왼쪽 손목
    {6, 8},   // 오른쪽 어깨 - 오른쪽 팔꿈치
    {8, 10},  // 오른쪽 팔꿈치 - 오른쪽 손목
    {5, 11},  // 왼쪽 어깨 - 왼쪽 엉덩이
    {6, 12},  // 오른쪽 어깨 - 오른쪽 엉덩이
    {11, 12}, // 엉덩이 - 엉덩이
    {11, 13}, // 왼쪽 엉덩이 - 왼쪽 무릎
    {13, 15}, // 왼쪽 무릎 - 왼쪽 발목
    {12, 14}, // 오른쪽 엉덩이 - 오른쪽 무릎
    {14, 16}  // 오른쪽 무릎 - 오른쪽 발목
};

PoseEstimator::PoseEstimator(const AppConfig& config) 
    : config_(config), 
      inputH(config.pose.input_height), 
      inputW(config.pose.input_width), 
      numKeypoints(17), // COCO 모델 기준, 필요 시 설정 가능
      batchSize(1),
      initialized_(false) // 초기화 플래그 false로 시작
{
    // CUDA 사용 여부 확인
    if (!config_.pose.use_cuda) {
        std::cerr << "[오류] PoseEstimator: config.yaml에서 use_cuda가 false로 설정되었습니다. TensorRT 모델은 CUDA가 필요합니다." << std::endl;
        return; // 초기화 실패
    }

    // 모델 로드 시 config에서 경로 사용
    if (!loadEngine(config_.pose.model_path)) {
        std::cerr << "TensorRT 엔진 로드 실패: " << config_.pose.model_path << std::endl;
        return;
    }
    
    // 입출력 인덱스 설정 - 모델의 실제 텐서 이름 사용
    inputIndex = engine->getBindingIndex("input.1");
    
    // 모델에 출력 텐서가 2개 있으므로, 첫 번째 출력 텐서 사용
    outputIndex = engine->getBindingIndex("onnx::Concat_2957");
    
    // 출력 텐서가 없으면 두 번째 출력 텐서 시도
    if (outputIndex == -1) {
        outputIndex = engine->getBindingIndex("2990");
    }
    
    if (inputIndex == -1 || outputIndex == -1) {
        std::cerr << "TensorRT 모델 바인딩 인덱스를 찾을 수 없습니다." << std::endl;
        std::cerr << "입력 인덱스: " << inputIndex << ", 출력 인덱스: " << outputIndex << std::endl;
        return;
    }
    
    // 출력 크기 계산
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputSize *= outputDims.d[i];
    }
    
    // TensorRT 엔진의 바인딩 수 확인 (1 입력 + 2 출력 = 3)
    int numBindings = engine->getNbBindings();
    
    // 모든 바인딩에 대해 메모리 할당
    for (int i = 0; i < numBindings; i++) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = batchSize;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        cudaMalloc(&buffers[i], size * sizeof(float));
    }
    
    // 호스트 버퍼 할당
    inputBufferHost = new float[batchSize * 3 * inputH * inputW];
    outputBufferHost = new float[batchSize * outputSize];
    
    initialized_ = true; // 모든 초기화 성공
}

PoseEstimator::~PoseEstimator() {
    if (!initialized_) return; // 초기화 실패 시 메모리 해제 건너뛰기
    
    // 모든 바인딩에 대해 메모리 해제
    int numBindings = engine ? engine->getNbBindings() : 0;
    for (int i = 0; i < numBindings; i++) {
        if (buffers[i]) {
            cudaFree(buffers[i]);
            buffers[i] = nullptr;
        }
    }
    
    // 호스트 버퍼 해제
    delete[] inputBufferHost;
    delete[] outputBufferHost;
    
    // 명시적으로 순서대로 소멸 (컨텍스트 -> 엔진 -> 런타임)
    context.reset();
    engine.reset();
    runtime.reset();
}

bool PoseEstimator::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file) {
        std::cerr << "TensorRT 엔진 파일을 열 수 없습니다: " << enginePath << std::endl;
        return false;
    }
    
    // 파일 크기 확인
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 모델 파일 로드
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    
    // TensorRT 런타임 생성 - 클래스 멤버 변수로 설정
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "TensorRT 런타임 생성 실패" << std::endl;
        return false;
    }
    
    // 엔진 생성
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    if (!engine) {
        std::cerr << "TensorRT 엔진 생성 실패" << std::endl;
        return false;
    }
    
    // 실행 컨텍스트 생성
    context.reset(engine->createExecutionContext());
    if (!context) {
        std::cerr << "TensorRT 실행 컨텍스트 생성 실패" << std::endl;
        return false;
    }
    
    return true;
}

bool PoseEstimator::detect(const cv::Mat& image, std::vector<std::vector<cv::Point>>& keypoints) {
    if (!initialized_) { // 초기화 확인 추가
        std::cerr << "[오류] PoseEstimator가 제대로 초기화되지 않았습니다." << std::endl;
        return false;
    }
    
    if (!engine || !context) { // 기존 엔진/컨텍스트 확인 유지
        std::cerr << "TensorRT 엔진이 초기화되지 않았습니다." << std::endl;
        return false;
    }
    
    // 이미지 전처리 (멤버 변수 사용)
    preprocess(image, inputBufferHost);
    
    // 입력 데이터 GPU로 복사 (멤버 변수 사용)
    cudaMemcpy(buffers[inputIndex], inputBufferHost, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);
    
    // 추론 실행
    context->executeV2(buffers);
    
    // 출력 데이터 CPU로 복사 - 제거됨 (멤버 변수 사용)
    cudaMemcpy(outputBufferHost, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 후처리를 통해 키포인트 추출 (멤버 변수 사용)
    postprocess(outputBufferHost, image.size(), keypoints);
    
    return true;
}

void PoseEstimator::preprocess(const cv::Mat& image, float* inputBuffer) {
    // cv::dnn::blobFromImage를 사용하여 전처리
    // 설정에서 mean, std 값 가져오기
    cv::Scalar mean(config_.pose.mean[0], config_.pose.mean[1], config_.pose.mean[2]);
    cv::Scalar std(config_.pose.std[0], config_.pose.std[1], config_.pose.std[2]);

    // 주의: blobFromImage는 BGR 순서로 처리. config.yaml의 주석과 코드 확인 필요
    // 현재 mean, std는 RGB 순서로 가정하고 로드했으므로, BGR 순서로 변경하여 사용
    cv::Scalar meanBGR(mean[2], mean[1], mean[0]);

    // inputW, inputH는 멤버 변수 사용
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0/255.0, cv::Size(inputW, inputH), meanBGR, true, false);
    
    // std도 BGR 순서에 맞게 사용
    cv::Scalar stdBGR(std[2], std[1], std[0]); 

    int channelLength = inputH * inputW;

    // blob 데이터를 inputBufferHost로 복사하며 표준편차 적용 (split/memcpy 대체)
    for(int c = 0; c < 3; ++c) { // c=0: Blue, c=1: Green, c=2: Red
        float* srcPtr = blob.ptr<float>(0, c); // NCHW 형식의 blob에서 채널 포인터 가져오기
        float* dstPtr = inputBuffer + c * channelLength;
        float stdVal = (float)stdBGR[c]; // 해당 채널의 표준편차

        // 0으로 나누기 방지
        if (std::abs(stdVal) < 1e-6) {
            stdVal = 1.0f; // 표준편차가 0에 가까우면 나누지 않음 (또는 에러 처리)
             std::cerr << "[Warning Preprocess] Standard deviation for channel " << c << " is close to zero." << std::endl;
        }

        for (int i = 0; i < channelLength; ++i) {
            dstPtr[i] = srcPtr[i] / stdVal;
        }
    }
}

void PoseEstimator::postprocess(float* outputBuffer, const cv::Size& originalSize, std::vector<std::vector<cv::Point>>& keypoints) {
    keypoints.clear();
    keypoints.resize(1); // 한 명의 사람만 가정
    keypoints[0].resize(numKeypoints);
    
    // 설정에서 히트맵 크기 가져오기
    int heatmapH = config_.pose.heatmap_height;
    int heatmapW = config_.pose.heatmap_width;
    int heatmapSize = heatmapH * heatmapW;

    for (int k = 0; k < numKeypoints; k++) {
        // 현재 키포인트(채널)의 히트맵 데이터에 대한 Mat 헤더 생성 (복사 없음)
        cv::Mat heatmap(heatmapH, heatmapW, CV_32FC1, outputBuffer + k * heatmapSize);
        
        // 히트맵에서 최대값과 위치 찾기
        double maxValDouble;
        cv::Point maxLoc;
        cv::minMaxLoc(heatmap, nullptr, &maxValDouble, nullptr, &maxLoc);
        float maxVal = static_cast<float>(maxValDouble);
        
        // 원본 이미지 크기로 스케일링
        float x = static_cast<float>(maxLoc.x) / heatmapW * originalSize.width;
        float y = static_cast<float>(maxLoc.y) / heatmapH * originalSize.height;
        
        // 신뢰도가 임계값보다 높으면 키포인트 추가 (설정값 사용)
        if (maxVal > config_.pose.confidence_threshold) {
            keypoints[0][k] = cv::Point(static_cast<int>(x), static_cast<int>(y));
        } else {
            keypoints[0][k] = cv::Point(-1, -1); // 신뢰도가 낮은 키포인트는 무효화
        }
    }
}

void PoseEstimator::drawKeypoints(cv::Mat& image, const std::vector<std::vector<cv::Point>>& keypoints) {
    if (keypoints.empty()) return;
    
    // 각 사람에 대해
    for (const auto& person : keypoints) {
        // 키포인트 그리기
        for (int i = 0; i < person.size(); i++) {
            if (person[i].x >= 0 && person[i].y >= 0) { // 유효한 키포인트만
                cv::circle(image, person[i], 5, colors[i], -1);
            }
        }
        
        // 스켈레톤 그리기 (키포인트 연결)
        for (const auto& limb : skeleton) {
            int i = limb.first;
            int j = limb.second;
            if (i < person.size() && j < person.size() &&
                person[i].x >= 0 && person[i].y >= 0 &&
                person[j].x >= 0 && person[j].y >= 0) {
                cv::line(image, person[i], person[j], cv::Scalar(255, 255, 255), 2);
            }
        }
    }
} 
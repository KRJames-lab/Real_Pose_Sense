#include "PoseEstimator.h"
#include <fstream>
#include <cuda_runtime_api.h>
#include <iostream>

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

PoseEstimator::PoseEstimator(const std::string& modelPath) 
    : inputH(512), inputW(512), numKeypoints(17), batchSize(1) {
    if (!loadEngine(modelPath)) {
        std::cerr << "TensorRT 엔진 로드 실패: " << modelPath << std::endl;
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
}

PoseEstimator::~PoseEstimator() {
    // 모든 바인딩에 대해 메모리 해제
    int numBindings = engine ? engine->getNbBindings() : 0;
    for (int i = 0; i < numBindings; i++) {
        if (buffers[i]) {
            cudaFree(buffers[i]);
            buffers[i] = nullptr;
        }
    }
    
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
    if (!engine || !context) {
        std::cerr << "TensorRT 엔진이 초기화되지 않았습니다." << std::endl;
        return false;
    }
    
    // 입력 버퍼 할당
    float* inputBuffer = new float[batchSize * 3 * inputH * inputW];
    
    // 이미지 전처리
    preprocess(image, inputBuffer);
    
    // 입력 데이터 GPU로 복사
    cudaMemcpy(buffers[inputIndex], inputBuffer, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);
    
    // 추론 실행
    context->executeV2(buffers);
    
    // 출력 데이터 CPU로 복사
    float* outputBuffer = new float[batchSize * outputSize];
    cudaMemcpy(outputBuffer, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 후처리를 통해 키포인트 추출
    postprocess(outputBuffer, image.size(), keypoints);
    
    // 메모리 해제
    delete[] inputBuffer;
    delete[] outputBuffer;
    
    return true;
}

void PoseEstimator::preprocess(const cv::Mat& image, float* inputBuffer) {
    // 입력 이미지 리사이즈 - 모델 입력 크기 512x512에 맞춤
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputW, inputH));
    
    // 이미지 정규화 (0-255 -> 0-1)
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0/255.0);
    
    // 평균, 표준편차로 정규화
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    
    // HWC -> CHW 형식으로 변환 및 버퍼에 복사
    int channelLength = inputH * inputW;
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < inputH; h++) {
            for (int w = 0; w < inputW; w++) {
                float pixel = normalized.at<cv::Vec3f>(h, w)[c];
                inputBuffer[c * channelLength + h * inputW + w] = (pixel - mean[c]) / std[c];
            }
        }
    }
}

void PoseEstimator::postprocess(float* outputBuffer, const cv::Size& originalSize, std::vector<std::vector<cv::Point>>& keypoints) {
    // 모델 출력에서 히트맵 추출하여 각 키포인트의 위치 결정
    keypoints.clear();
    keypoints.resize(1); // 한 명의 사람만 가정
    keypoints[0].resize(numKeypoints);
    
    // 출력 형식에 따라 구현 (Higher-HRNet 모델의 경우 출력 크기가 다름)
    int heatmapH = 128; // onnx::Concat_2957의 경우 128x128
    int heatmapW = 128;
    
    for (int k = 0; k < numKeypoints; k++) {
        // 히트맵에서 최대값 찾기
        float maxVal = -1;
        int maxH = -1, maxW = -1;
        
        for (int h = 0; h < heatmapH; h++) {
            for (int w = 0; w < heatmapW; w++) {
                int idx = k * heatmapH * heatmapW + h * heatmapW + w;
                if (outputBuffer[idx] > maxVal) {
                    maxVal = outputBuffer[idx];
                    maxH = h;
                    maxW = w;
                }
            }
        }
        
        // 원본 이미지 크기로 스케일링
        float x = static_cast<float>(maxW) / heatmapW * originalSize.width;
        float y = static_cast<float>(maxH) / heatmapH * originalSize.height;
        
        // 신뢰도가 임계값보다 높으면 키포인트 추가
        if (maxVal > 0.3) {
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
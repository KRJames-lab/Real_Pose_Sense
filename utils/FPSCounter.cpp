#include "FPSCounter.h"

namespace Utils {
    FPSCounter::FPSCounter() {
        lastTime = std::chrono::high_resolution_clock::now();
    }

    float FPSCounter::update() {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float delta = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // 프레임 시간 기록
        frameTimes.push_back(delta);
        
        // 최대 샘플 수 유지
        if (frameTimes.size() > maxSamples) {
            frameTimes.pop_front();
        }
        
        // 평균 프레임 시간 계산
        float avgTime = 0.0f;
        for (float t : frameTimes) {
            avgTime += t;
        }
        avgTime /= frameTimes.size();
        
        // FPS 계산 및 반환
        return 1.0f / avgTime;
    }
} 
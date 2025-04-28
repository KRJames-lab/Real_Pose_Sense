#pragma once

#include <chrono>
#include <deque>

namespace Utils {
    // FPS 측정을 위한 클래스
    class FPSCounter {
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
        std::deque<float> frameTimes;
        const int maxSamples = 30; // 이동 평균을 위한 샘플 수
        
    public:
        FPSCounter();
        float update();
    };
} 
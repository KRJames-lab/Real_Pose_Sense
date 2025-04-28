#pragma once

#include <opencv2/opencv.hpp>

namespace Utils {
    class KeyboardHandler {
    public:
        KeyboardHandler();
        
        // 키 입력 처리
        char waitKey(int delay = 1);
        
        // 종료 키가 눌렸는지 확인
        bool isQuitPressed() const;
        
        // 저장 키가 눌렸는지 확인
        bool isSavePressed() const;
        
        // 마지막 키 가져오기
        char getLastKey() const;
        
    private:
        char lastKey;
    };
} 
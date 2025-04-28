#pragma once

#include <string>
#include <ctime>
#include <sys/stat.h>

namespace Utils {
    namespace FileUtils {
        // 디렉토리 존재 확인 함수
        bool directoryExists(const std::string& dir);
        
        // 디렉토리 생성 함수
        bool createDirectory(const std::string& dir);
        
        // 현재 시간을 문자열로 반환하는 함수
        std::string getTimestamp();
        
        // 다음 사용 가능한 result 폴더 번호 찾기
        int findNextResultFolder(const std::string& baseDir);
    } 
} 
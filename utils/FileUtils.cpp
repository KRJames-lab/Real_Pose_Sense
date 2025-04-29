#include "FileUtils.h"
#include <iostream>
#include <sstream>

namespace Utils {
    namespace FileUtils {

        bool directoryExists(const std::string& dir) {
            struct stat info;
            return stat(dir.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
        }

        bool createDirectory(const std::string& dir) {
            if (directoryExists(dir)) {
                return true; 
            }
            
            #ifdef _WIN32
                return _mkdir(dir.c_str()) == 0;
            #else
                return mkdir(dir.c_str(), 0755) == 0;
            #endif
        }

        int findNextResultFolder(const std::string& baseDir) {
            int folderNum = 1;
            
            while (true) {
                std::stringstream ss;
                ss << baseDir << "result" << folderNum;
                std::string folderPath = ss.str();
                
                if (!directoryExists(folderPath)) {
                    return folderNum;
                }
                
                folderNum++;
            }
        }

    } 
} 
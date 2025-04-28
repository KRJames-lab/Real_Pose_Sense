#include "KeyboardHandler.h"

namespace Utils {
    KeyboardHandler::KeyboardHandler() : lastKey(0) {
    }

    char KeyboardHandler::waitKey(int delay) {
        lastKey = cv::waitKey(delay);
        return lastKey;
    }

    bool KeyboardHandler::isQuitPressed() const {
        return lastKey == 'q';
    }

    bool KeyboardHandler::isSavePressed() const {
        return lastKey == 's';
    }

    char KeyboardHandler::getLastKey() const {
        return lastKey;
    }
} 
//g++ -o main main.cpp -lrealsense2 `pkg-config --cflags --libs opencv4`
#include <iostream>     
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>  
#include <fstream>             
#include <ctime>               
#include <string>              
#include <sys/stat.h>          
#include <iomanip>             
#include <algorithm>           
#include <chrono>              
#include <deque>               
#include <thread>              
#include <sys/select.h>        
#include <unistd.h>            

// 설정 정보를 저장하는 구조체
struct AppConfig {
    struct {
        bool direct_conversion;
        
        struct {
            double clip_limit;
            int tile_grid_size;
        } clahe;
    } visualization;
    
    struct {
        struct {
            int width;
            int height;
            std::string format;
            int fps;
        } color, depth;
    } stream;
    
    struct {
        float min;
        float max;
    } depth_range;
    
    struct {
        std::string directory;
    } save;
};

// 설정 파일을 로드하는 함수
bool loadConfig(const std::string& config_file, AppConfig& config) {
    try {
        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "설정 파일을 열 수 없습니다: " << config_file << std::endl;
            return false;
        }
        
        fs["visualization"]["direct_conversion"] >> config.visualization.direct_conversion;
        fs["visualization"]["clahe"]["clip_limit"] >> config.visualization.clahe.clip_limit;
        fs["visualization"]["clahe"]["tile_grid_size"] >> config.visualization.clahe.tile_grid_size;

        fs["stream"]["color"]["width"] >> config.stream.color.width;
        fs["stream"]["color"]["height"] >> config.stream.color.height;
        fs["stream"]["color"]["format"] >> config.stream.color.format;
        fs["stream"]["color"]["fps"] >> config.stream.color.fps;
        
        fs["stream"]["depth"]["width"] >> config.stream.depth.width;
        fs["stream"]["depth"]["height"] >> config.stream.depth.height;
        fs["stream"]["depth"]["format"] >> config.stream.depth.format;
        fs["stream"]["depth"]["fps"] >> config.stream.depth.fps;

        fs["depth_range"]["min"] >> config.depth_range.min;
        fs["depth_range"]["max"] >> config.depth_range.max;

        fs["save"]["directory"] >> config.save.directory;
        
        fs.release();
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "설정 파일 로드 오류: " << e.what() << std::endl;
        return false;
    }
}

// 디렉토리 존재 확인 함수
bool directory_exists(const std::string& dir) {
    struct stat info;
    return stat(dir.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

// 디렉토리 생성 함수
bool create_directory(const std::string& dir) {
    if (directory_exists(dir)) {
        return true; 
    }
    
    #ifdef _WIN32
        return _mkdir(dir.c_str()) == 0;
    #else
        return mkdir(dir.c_str(), 0755) == 0;
    #endif
}

// 현재 시간을 문자열로 반환하는 함수
std::string get_timestamp()
{
    std::time_t now = std::time(nullptr);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(buffer);
}

// FPS 측정을 위한 클래스
class FPSCounter {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time;
    std::deque<float> frame_times;
    const int max_samples = 30; // 이동 평균을 위한 샘플 수
    
public:
    FPSCounter() {
        last_time = std::chrono::high_resolution_clock::now();
    }
    
    float update() {
        auto current_time = std::chrono::high_resolution_clock::now();
        float delta = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;
        
        // 프레임 시간 기록
        frame_times.push_back(delta);
        
        // 최대 샘플 수 유지
        if (frame_times.size() > max_samples) {
            frame_times.pop_front();
        }
        
        // 평균 프레임 시간 계산
        float avg_time = 0.0f;
        for (float t : frame_times) {
            avg_time += t;
        }
        avg_time /= frame_times.size();
        
        // FPS 계산 및 반환
        return 1.0f / avg_time;
    }
};

// 깊이 이미지 시각화 함수 - 설정에 따라 변환 방식 선택
cv::Mat enhanced_depth_visualization(const rs2::depth_frame& depth_frame, const AppConfig& config)
{
    float min_depth = config.depth_range.min;
    float max_depth = config.depth_range.max;
    bool direct_conversion = config.visualization.direct_conversion;
    
    // 깊이 프레임에서 데이터 가져오기
    int width = depth_frame.get_width();
    int height = depth_frame.get_height();
    
    cv::Mat depth_8bit;
    
    if (direct_conversion) {
        // 32비트 float -> 8비트 변환 방식
        cv::Mat depth_float(height, width, CV_32FC1);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth_value = depth_frame.get_distance(x, y);
                
                if (depth_value < min_depth || depth_value > max_depth || depth_value <= 0) {
                    depth_float.at<float>(y, x) = 0.0f;
                } else {
                    depth_float.at<float>(y, x) = depth_value;
                }
            }
        }
        
        // 깊이 값 정규화 (min_depth ~ max_depth -> 0.0 ~ 1.0)
        cv::Mat normalized_depth;
        cv::subtract(depth_float, min_depth, normalized_depth);
        cv::divide(normalized_depth, (max_depth - min_depth), normalized_depth);
        
        // 정규화된 float 이미지를 8비트로 직접 변환 (0.0~1.0 -> 0~255)
        normalized_depth.convertTo(depth_8bit, CV_8UC1, 255.0);
    } else {
        // 32비트 float -> 16비트 -> 8비트 변환
        cv::Mat depth_image(height, width, CV_16UC1);
        
        // 깊이 데이터를 16비트(0-65535) 범위로 정규화
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth_value = depth_frame.get_distance(x, y);
                
                if (depth_value < min_depth || depth_value > max_depth || depth_value <= 0) {
                    depth_image.at<ushort>(y, x) = 0;
                } else {
                    // 깊이값을 0-65535 범위로 변환 (16비트)
                    depth_image.at<ushort>(y, x) = static_cast<ushort>((depth_value - min_depth) / (max_depth - min_depth) * 65535);
                }
            }
        }
        
        // 16비트에서 8비트로 변환
        depth_image.convertTo(depth_8bit, CV_8UC1, 1.0/256.0);
    }
    
    // CLAHE(Contrast Limited Adaptive Histogram Equalization)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(config.visualization.clahe.clip_limit);
    clahe->setTilesGridSize(cv::Size(config.visualization.clahe.tile_grid_size, 
                                     config.visualization.clahe.tile_grid_size));
    
    cv::Mat enhanced_depth;
    clahe->apply(depth_8bit, enhanced_depth);
    
    // 히트맵으로 변환
    cv::Mat colormap;
    cv::applyColorMap(enhanced_depth, colormap, cv::COLORMAP_TURBO);
    
    // 범위 정보를 이미지에 추가
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << min_depth << "m ~ " << max_depth << "m";
    cv::putText(colormap, ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // 중앙 지점의 거리 정보 추가
    float center_dist = depth_frame.get_distance(depth_frame.get_width() / 2, depth_frame.get_height() / 2);
    std::stringstream center_ss;
    center_ss << std::fixed << std::setprecision(2) << "Distance to center(m): " << center_dist << "m";
    cv::putText(colormap, center_ss.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    return colormap;
}

// 깊이 맵을 바이너리 파일로 저장하는 함수
void save_depth_to_bin(const rs2::depth_frame& depth_frame, const std::string& filename)
{
    // 깊이 프레임에서 데이터 가져오기
    int width = depth_frame.get_width();
    int height = depth_frame.get_height();
    
    // 바이너리 파일 열기
    std::ofstream outfile(filename, std::ios::binary);
    
    if (!outfile) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }
    
    // 파일 헤더 (너비와 높이 저장)
    outfile.write(reinterpret_cast<const char*>(&width), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&height), sizeof(int));
    
    // 모든 픽셀의 깊이 값을 저장
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float depth_value = depth_frame.get_distance(x, y);
            outfile.write(reinterpret_cast<const char*>(&depth_value), sizeof(float));
        }
    }
    
    outfile.close();
    std::cout << "깊이 맵이 저장되었습니다: " << filename << std::endl;
}

// 다음 사용 가능한 result 폴더 번호 찾기
int find_next_result_folder(const std::string& base_dir) {
    int folder_num = 1;
    
    while (true) {
        std::stringstream ss;
        ss << base_dir << "result" << folder_num;
        std::string folder_path = ss.str();
        
        if (!directory_exists(folder_path)) {
            return folder_num;
        }
        
        folder_num++;
    }
}

int main(int argc, char *argv[]) try
{
    std::string executable_path = argv[0];
    std::string executable_dir = executable_path.substr(0, executable_path.find_last_of("/\\"));
    std::string config_file = executable_dir + "/config.yaml";
    
    // 설정 로드
    AppConfig config;
    if (!loadConfig(config_file, config)) {
        std::cerr << "기본 설정을 사용합니다." << std::endl;
        
        // 기본 설정값 설정
        config.visualization.direct_conversion = true;
        config.visualization.clahe.clip_limit = 3.0;
        config.visualization.clahe.tile_grid_size = 16;
        
        config.stream.color.width = 640;
        config.stream.color.height = 480;
        config.stream.color.format = "BGR8";
        config.stream.color.fps = 30;
        
        config.stream.depth.width = 640;
        config.stream.depth.height = 480;
        config.stream.depth.format = "Z16";
        config.stream.depth.fps = 30;
        
        config.depth_range.min = 0.1f;
        config.depth_range.max = 1.0f;
        
        config.save.directory = "./results/";
    }
    
    // 결과 저장 디렉토리 설정 및 생성
    std::string results_dir = config.save.directory;
    if (!create_directory(results_dir)) {
        std::cerr << "결과 디렉토리 생성 실패: " << results_dir << std::endl;
        return EXIT_FAILURE;
    }
    
    // 현재 result 폴더 번호 (저장 시 증가)
    int result_folder_num = find_next_result_folder(results_dir);
    
    // RealSense 파이프라인 초기화
    rs2::pipeline p;      
    rs2::config cfg;      
    
    // 설정 파일에서 스트림 포맷 변환
    rs2_format color_format = RS2_FORMAT_BGR8;  
    if (config.stream.color.format == "RGB8") color_format = RS2_FORMAT_RGB8;
    else if (config.stream.color.format == "RGBA8") color_format = RS2_FORMAT_RGBA8;
    else if (config.stream.color.format == "BGRA8") color_format = RS2_FORMAT_BGRA8;
    
    rs2_format depth_format = RS2_FORMAT_Z16;  

    cfg.enable_stream(RS2_STREAM_COLOR, 
                     config.stream.color.width, 
                     config.stream.color.height, 
                     color_format, 
                     config.stream.color.fps);
    
    cfg.enable_stream(RS2_STREAM_DEPTH, 
                     config.stream.depth.width, 
                     config.stream.depth.height, 
                     depth_format, 
                     config.stream.depth.fps);
    
   
    p.start(cfg);       
    
    std::cout << "RealSense 카메라 시작됨. 's'를 누르면 이미지와 깊이 맵을 저장하고, 'q'를 누르면 종료합니다." << std::endl;
    std::cout << "파일은 " << results_dir << "resultN/ 디렉토리에 저장됩니다." << std::endl;
    
    std::cout << "====== 현재 설정 ======" << std::endl;
    
    std::cout << "[시각화 설정]" << std::endl;
    std::cout << "  - 변환 방식: " << (config.visualization.direct_conversion ? "직접 변환 (32비트->8비트)" : "단계별 변환 (32비트->16비트->8비트)") << std::endl;
    std::cout << "  - CLAHE 설정:" << std::endl;
    std::cout << "    * Clip Limit: " << config.visualization.clahe.clip_limit << std::endl;
    std::cout << "    * Tile Grid Size: " << config.visualization.clahe.tile_grid_size << "x" << config.visualization.clahe.tile_grid_size << std::endl;
    
    std::cout << "[스트림 설정]" << std::endl;
    std::cout << "  - 컬러 스트림: " << config.stream.color.width << "x" << config.stream.color.height 
              << ", 포맷: " << config.stream.color.format << ", FPS: " << config.stream.color.fps << std::endl;
    std::cout << "  - 깊이 스트림: " << config.stream.depth.width << "x" << config.stream.depth.height 
              << ", 포맷: " << config.stream.depth.format << ", FPS: " << config.stream.depth.fps << std::endl;
    
    std::cout << "[깊이 범위 설정]" << std::endl;
    std::cout << "  - 최소 깊이: " << config.depth_range.min << "m" << std::endl;
    std::cout << "  - 최대 깊이: " << config.depth_range.max << "m" << std::endl;
    
    std::cout << "[저장 설정]" << std::endl;
    std::cout << "  - 저장 디렉토리: " << config.save.directory << std::endl;
    
    std::cout << "======================" << std::endl;
    
    FPSCounter fps_counter;
    
    // 시각화 창 생성
    cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Enhanced Depth", cv::WINDOW_AUTOSIZE);
    
    std::cout << "'s'를 눌러서 저장하고, 'q'를 눌러서 종료하세요." << std::endl;
    
    // 키 입력 처리
    char key = 0;
    while(key != 'q') {
        // FPS 업데이트
        float fps = fps_counter.update();
        
        rs2::frameset frames = p.wait_for_frames();

        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();
        
        if (!depth || !color) {
            std::cerr << "유효하지 않은 프레임 발견. 건너뜁니다." << std::endl;
            continue;
        }
        
        auto width = depth.get_width();
        auto height = depth.get_height();

        // 컬러 이미지를 OpenCV 형식으로 변환
        cv::Mat color_image(cv::Size(width, height), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        
        // 깊이 맵 시각화
        cv::Mat enhanced_depth = enhanced_depth_visualization(depth, config);
        
        // FPS 정보 추가
        std::stringstream fps_ss;
        fps_ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(color_image, fps_ss.str(), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(enhanced_depth, fps_ss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        // 컨트롤 정보 추가
        cv::putText(color_image, "s: Save, q: Quit", cv::Point(10, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        
        // 이미지 표시
        cv::imshow("Color Image", color_image);
        cv::imshow("Enhanced Depth", enhanced_depth);
        
        // 키 입력 대기 (1ms)
        key = cv::waitKey(1);
        
        // 's' 키를 누르면 이미지와 깊이 맵 저장
        if (key == 's') {
            std::stringstream folder_ss;
            folder_ss << results_dir << "result" << result_folder_num << "/";
            std::string result_folder = folder_ss.str();
            
            if (!create_directory(result_folder)) {
                std::cerr << "폴더 생성 실패: " << result_folder << std::endl;
                continue;
            }
            
            std::string color_filename = result_folder + "color.png";
            std::string depth_colormap_filename = result_folder + "depth_colormap.png";
            std::string depth_bin_filename = result_folder + "depth.bin";
            
            cv::imwrite(color_filename, color_image);
            cv::imwrite(depth_colormap_filename, enhanced_depth);
            save_depth_to_bin(depth, depth_bin_filename);
            
            std::cout << "파일이 저장되었습니다. 폴더: result" << result_folder_num << std::endl;
            
            result_folder_num++;
        }
    }
    
    cv::destroyAllWindows();
    
    return EXIT_SUCCESS; 
}
catch (const rs2::error & e) 
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE; 
}
catch (const std::exception& e) 
{
    std::cerr << e.what() << std::endl; 
    return EXIT_FAILURE;  
}
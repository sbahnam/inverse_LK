#include <opencv2/opencv.hpp>

int pyramid_levels = 4;
int pyramid_scale = 2;

// Pyramids for previous and current image
std::vector<cv::Mat> prev_cam0_pyramid_;
std::vector<cv::Mat> prev_cam0_pyramid_restore;
std::vector<cv::Mat> curr_cam0_pyramid_;
std::vector<cv::Mat> curr_cam0_pyramid_restore;

void ScaleDown(const cv::Mat&, cv::Mat&);
void GenerateImagePyramid(const cv::Mat&, std::vector<cv::Mat>&);

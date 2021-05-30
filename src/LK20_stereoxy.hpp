#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


void TrackLKInv_stereoXY(const std::vector<cv::Mat>& vm_image_pyramid, const std::vector<cv::Mat>& vm_current_image_pyramid,
      const std::vector<cv::Point2f> &prev_points, std::vector<cv::Point2f> &curr_points,
      std::vector<unsigned char> &inlier_markers, const float error_threshold,
      const cv::Size winsize, const int max_iter, const float epsilon, const float dx_pred);

//Pre computation of inverse compositional LK
cv::Mat ComputeInvHessianstereo_XY(const cv::Mat& m_j);
cv::Mat ComputeImageGradient_stereoXY(const cv::Mat& _image);

// Loop functions
// Start loop
void ProcessInLayer_stereoXY(cv::Point2f& delta, const cv::Point2f& x0y0,
                    float& error,
                    const int& max_iter,
                    const float& epsilon,
                    const cv::Size& winsize,
                    const cv::Mat& m_cur_image_in_pyramid,
                    const cv::Mat& m_ref_image_in_pyramid,
                    const int& level,
                    const cv::Mat& hessian,
                    const cv::Mat& J);

float ComputeResiduals_stereoXY(cv::Mat& m_redisuals, 
                        const cv::Mat& cur_image, 
                        const cv::Mat& ref_image,
                        const int& level);
cv::Mat ComputeUpdateParams_stereoXY(const cv::Mat& m_hessian, 
                            const cv::Mat& m_J,
                            const cv::Mat& m_residuals);


bool WarpCurrentImage_stereoXY(const cv::Mat& src, cv::Mat &dst, const float& dx_i, const float& dy_i,
                        const cv::Point2f& x0y0, const cv::Size& winsize);

cv::Mat warp_interpolater_stereoXY(const cv::Mat& src, const float& x0, const float& y0, const cv::Size& winsize);



#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


void TrackLKInv_stereo(const std::vector<cv::Mat>& vm_image_pyramid, const std::vector<cv::Mat>& vm_current_image_pyramid,
      const std::vector<cv::Point2f> &prev_points, std::vector<cv::Point2f> &curr_points,
      std::vector<unsigned char> &inlier_markers, const float error_threshold,
      const cv::Size winsize, const int max_iter, const float epsilon, const float dx_pred);

//Pre computation of inverse compositional LK
float ComputeInvHessian_stereoX(const cv::Mat& m_j);
cv::Mat ComputeImageGradient_stereoX(const cv::Mat& _image);

// Loop functions
// Start loop
void ProcessInLayer_stereoX(float& delta_x, const cv::Point2f& x0y0,
                                float& error,
                                const int& max_iter,
                                const float& epsilon,
                                const cv::Size& winsize,
                                const cv::Mat& m_cur_image_in_pyramid,
                                const cv::Mat& m_ref_image_in_pyramid,
                                const int& level,
                                const float& m_inv_hessian,
                                const cv::Mat& m_J);

float ComputeResiduals_stereoX(cv::Mat& m_redisuals, 
                        const cv::Mat& cur_image, 
                        const cv::Mat& ref_image,
                        const int& level);
float ComputeUpdateParams_stereoX(const float& m_inv_hessian, 
                            const cv::Mat& m_J,
                            const cv::Mat& m_residuals);


bool WarpCurrentImage_stereoX(const cv::Mat& src, cv::Mat &dst, const float& dx_i,
                        const cv::Point2f& x0y0, const cv::Size& winsize);

cv::Mat warp_interpolater_stereoX(const cv::Mat& src, const float& x0, const float& y0, const cv::Size& winsize);
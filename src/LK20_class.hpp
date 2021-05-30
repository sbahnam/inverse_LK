#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class invLK {
  public:
    invLK(const int pyr_size, const float pyr_scale, const cv::Size img_size, cv::Size win_size); // constructor
    // Track features from previous image to new image
    void TrackLKInv(const std::vector<cv::Mat>& vm_image_pyramid, const std::vector<cv::Mat>& vm_current_image_pyramid,
                    const std::vector<cv::Point2f> &prev_points, std::vector<cv::Point2f> &curr_points,
                    std::vector<unsigned char> &inlier_markers, const float error_threshold,
                    const int max_iter, const float epsilon, const float max_distance);

  private:
    // PRE COMPUTATION
    // Warps the previous image used as reference warp (+ boundaries for gradient calculation)
    void warp_interpolater(const cv::Mat& src, int& l, const float& x0, const float& y0);

    // Computes image gradient of the previous warped image with -1 0 1 kernel
    void ComputeImageGradient(const int& l);

    // Computes inverse Hessian of previous warped image
    void ComputeInvHessian(const int& l);


    // LOOP COMPUTATION
    // Start loop
    void ProcessInLayer(const int& max_iter, const float& epsilon,
                        const cv::Mat& m_cur_image_in_pyramid, const int& level);

    // Calculate error and creates residual matrix (current warp - previous warp)
    void ComputeResiduals(cv::Mat& m_redisuals, const cv::Mat& cur_image, const int& level);

    // Updates window translation parameters
    void ComputeUpdateParams(const cv::Mat& m_residuals, const int& l, cv::Point2f& dxdxy);

    // Warps the current image using the parameters of the sliding window
    // @todo: make WarpCurrentImage and warp_interpolater same function
    bool WarpCurrentImage(const cv::Mat& src, cv::Mat &dst, const float& dx_i, const float& dy_i, const cv::Point2f& x0y0);



    // private variables
    int m_pyramid_level; // Input - level of image pyramid
    float scale_level = 1.f; // Input - scale change between levels in image pyramid
    cv::Size winsize; // Input - size of the sliding window
    float sq_error; // Sum squared error of the residual matrix
    cv::Point2f delta; // Movement w.r.t. estimate position (curr[pt]) on scale

    std::vector<cv::Mat1f> mvm_ref_image_pyramid; // previous warped images
    std::vector<cv::Mat1f> mvm_J; // matrix of image gradient of previous warped image on all pyramid level
    std::vector<cv::Mat1f> mvm_inv_hessian; // inverse hessian 2x2 matrix of mvm_J on all pyramid level
    std::vector<cv::Point2f> mm_H0s; // current (predicted) feature location scaled on all pyramid levels
};
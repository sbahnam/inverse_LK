#include "LK20_stereo.hpp"

cv::Mat ComputeImageGradient_stereoX(const cv::Mat& _image) {
  cv::Mat1f m_dxdy = cv::Mat::zeros((_image.rows-1)*(_image.cols-1), 1, CV_32F);
  // if (m_dxdy.isContinuous() && _image.isContinuous())
  // {
    static int imCols = _image.cols;

    static int i,j;
    static float* p;
    static const float* q;
    p = m_dxdy.ptr<float>(0);
    q = _image.ptr<float>(0);
    static int k;
    k = 0;
    for( i = 0; i < _image.rows-1; ++i)
    {
        for ( j = 0; j < imCols-1; ++j)
        {
          if (j>0)
            p[k] = q[k + i + 1] - q[k + i -1];
          else
            p[k] = q[k + i + 1] - q[k + i];
          k++;
        }
    }
  // }
  // else
  // {
  //   std::cout<<"ERROR::ComputeImageGradient_stereoX:: cv mat not aligned"<<std::endl;
  // }
  return m_dxdy;
}

float ComputeInvHessian_stereoX(const cv::Mat& m_j)
{
  float m_hessian = 0.f;

  // if (m_j.isContinuous())
  // {
    static int mjCols = m_j.cols;

    static const float* q;
    q = m_j.ptr<float>(0);
    static int i;
    for( i = 0; i <  m_j.rows; ++i)
    {
      m_hessian += q[i] * q[i];
    }
  // }
  // else
  // {
  //   std::cout<<"ERROR::ComputeInvHessian_stereoX:: cv mat not algined"<<std::endl;
  // }
  return 1.f/m_hessian;
}

float ComputeResiduals_stereoX(cv::Mat& m_residuals,
                                  const cv::Mat& cur_image,
                                  const cv::Mat& ref_image)
{
  static float f_residual;
  f_residual = 0.f;

  // if (m_residuals.isContinuous() && cur_image.isContinuous() && ref_image.isContinuous())
  // {
    static float* p;
    static const float* q;
    static const float* r;
    p = m_residuals.ptr<float>(0);
    q = cur_image.ptr<float>(0);
    r = ref_image.ptr<float>(0);
    static int i, j, k;
    k = 0;
    for( i = 0; i < cur_image.rows; ++i)
    {
        for (j = 0; j < cur_image.cols; ++j)
        {
          p[k] = q[k] - r[k + i];
          f_residual += p[k]*p[k];
          k++;
        }
    }
  // }
  // else
  // {
  //   std::cout<<"ERROR::ComputeResiduals_stereoX:: cv mat not aligned"<<std::endl;
  // }

  return sqrt(f_residual/(float)(cur_image.rows*cur_image.cols));
}


float ComputeUpdateParams_stereoX(const float& m_inv_hessian, 
                                        const cv::Mat& m_J,
                                        const cv::Mat& m_residuals)
{
  static float dp1;
  dp1 = 0.f;

  // if (m_residuals.isContinuous() && m_J.isContinuous())
  // {
    static const float* q;
    static const float* r;
    q = m_J.ptr<float>(0);
    r = m_residuals.ptr<float>(0);
    static int i;
    for(i = 0; i < m_residuals.rows * m_residuals.cols; ++i)
    {
      dp1 += q[i]*r[i];
    }
  // }
  // else
  // {
  //   std::cout<<"ERROR::ComputeUpdateParams_stereoX: cv mat not aligned"<<std::endl;
  // }

  return m_inv_hessian * dp1;
}

void ProcessInLayer_stereoX(float& delta_x, const cv::Point2f& x0y0,
                                float& error,
                                const int& max_iter,
                                const float& epsilon,
                                const cv::Size& winsize,
                                const cv::Mat& m_cur_image_in_pyramid,
                                const cv::Mat& m_ref_image_in_pyramid,
                                const int& level,
                                const float& m_inv_hessian,
                                const cv::Mat& m_J)
{
  static cv::Mat1f m_residuals = cv::Mat::zeros(winsize.height*winsize.width, 1, CV_32F);
  static cv::Mat1f m_cur_working_image = cv::Mat::zeros(winsize.height, winsize.width, CV_32F);
  static float m_update_param;
  static float dx_i;
  static bool success_warp;
  static int argmin;
  static std::vector<float> errors_lst;
  static std::vector<float> delta_xs;
  errors_lst.clear();
  delta_xs.clear();
  dx_i = delta_x;
  // Inverse Compositional Algorithm
  for(int itr = 0; itr < max_iter; itr++)
  {
    success_warp = WarpCurrentImage_stereoX(m_cur_image_in_pyramid, m_cur_working_image, dx_i, x0y0, winsize);
    if (success_warp != true) break; // This happens if part of the window goes outside of the image (no border used)
    error = ComputeResiduals_stereoX(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
    errors_lst.push_back(error);
    delta_xs.push_back(dx_i);
    // fprintf(stderr,"Itr[%d|%d]: %.6f\n",itr,level,error);
    // std::cout<<dx_i<<", "<<m_update_param<<std::endl;
    m_update_param = ComputeUpdateParams_stereoX(m_inv_hessian, m_J, m_residuals);
    dx_i -= m_update_param;
    
    if(abs(m_update_param)  < epsilon) break;
  }

  if (errors_lst.size()>0) // in case first window is already outside the image
  {
    argmin = std::min_element(errors_lst.begin(), errors_lst.end()) - errors_lst.begin();
    delta_x = delta_xs[argmin];
    error = errors_lst[argmin];
  }
  return;
}

void TrackLKInv_stereo(const std::vector<cv::Mat>& vm_image_pyramid, const std::vector<cv::Mat>& vm_current_image_pyramid,
                      const std::vector<cv::Point2f>& prev_points, std::vector<cv::Point2f> &curr_points,
                      std::vector<unsigned char> &inlier_markers, const float error_threshold,
                      const cv::Size winsize, const int max_iter, const float epsilon, const float dx_pred)
{
  static std::vector<cv::Mat1f> mvm_J;
  static std::vector<float> mvm_inv_hessian;
  static std::vector<cv::Point2f> mm_H0s;

  static int m_pyramid_level;
  static float error;
  std::vector<cv::Mat1f> mvm_ref_image_pyramid;
  static float float_prev_rect_scaled_x;
  static float float_prev_rect_scaled_y;
  static float delta_x; // Movement w.r.t. estimate position (curr[pt]) on scale

  static float scaled;
  static float scale_level;
  if (m_pyramid_level > 1)
    scale_level = vm_current_image_pyramid[0].cols/vm_current_image_pyramid[1].cols;
  else scale_level = 1;

  for (int pt=0; pt<prev_points.size(); pt++)
  {
    m_pyramid_level = vm_image_pyramid.size(); //re assign pyramid level (in case previous point was done on lower pyramid size)
    error = 9999.f; // should be initialized higher than error_threshold in case the window is outside the image for all scales
    mvm_J.clear(); //@todo: intialize pyramids with correct cv mat structures and re-use them.
    mvm_J.reserve(m_pyramid_level);
    mvm_inv_hessian.clear();
    mvm_inv_hessian.reserve(m_pyramid_level);
    mvm_ref_image_pyramid.clear(); // Image regions on all scales of prev image used as reference (target window)
    mm_H0s.clear();
    mm_H0s.reserve(m_pyramid_level);

    for (int l = 0; l < m_pyramid_level; l++)
    {
      if (l==0)
      {
        float_prev_rect_scaled_x = prev_points[pt].x - (winsize.width - 1)/2.f;
        float_prev_rect_scaled_y =prev_points[pt].y - (winsize.height - 1)/2.f;
      }
      else
      {
        scaled = vm_image_pyramid[0].cols/vm_image_pyramid[l].cols;
        float_prev_rect_scaled_x = (prev_points[pt].x - (scaled-1.f)/2.f)/scaled - (winsize.width-1)/2.f;
        float_prev_rect_scaled_y = (prev_points[pt].y - (scaled-1.f)/2.f)/scaled - (winsize.height-1)/2.f;
      }
      if (float_prev_rect_scaled_x >= 0 && float_prev_rect_scaled_y >= 0 &&             // @todo: check curr as well here, so you can discard some of them and do not require to do that later
            float_prev_rect_scaled_x + winsize.width + 2 < vm_image_pyramid[l].cols && // +2 because image gradient uses
            float_prev_rect_scaled_y + winsize.height + 2 < vm_image_pyramid[l].rows)   // +1 to calculate values at border
                                                                                        // and +1 for interpolation
      {
        //@ todo: add one more pixel in x and y. To do a -1 0 1 gradient calculation for x and y = 0 as well
        mvm_ref_image_pyramid.push_back(warp_interpolater_stereoX(vm_image_pyramid[l],
                                                        float_prev_rect_scaled_x,
                                                        float_prev_rect_scaled_y,
                                                        winsize));
      }
      else break;
      
      if (l==0)
      mm_H0s.push_back(cv::Point2f(curr_points[pt].x-(winsize.width - 1)/2.f, 
                        curr_points[pt].y-(winsize.height - 1)/2.f));
      else
      {
        scaled = vm_image_pyramid[0].cols/vm_image_pyramid[l].cols;
        mm_H0s.push_back(cv::Point2f((curr_points[pt].x - (scaled-1.f)/2.f)/scaled - (winsize.width -1)/2.f,
                                  (curr_points[pt].y - (scaled-1.f)/2.f)/scaled - (winsize.height -1)/2.f ));
      }
      // Pre compute gradient and hessian on each pyramid (inverse compositional method)
      mvm_J.push_back(ComputeImageGradient_stereoX(mvm_ref_image_pyramid[l]));
      mvm_inv_hessian.push_back(ComputeInvHessian_stereoX(mvm_J[l]));
    }

    delta_x =0.f;
    m_pyramid_level = mvm_ref_image_pyramid.size(); // in case that the bigger pyramid levels are outside of the image

    // Coarse-to-Fine Optimization 
    for (int level = 0; level < m_pyramid_level; level++) {
      // std::cout<<mvm_ref_image_pyramid[m_pyramid_level - level -1]<<std::endl<<std::endl;
      // std::cout<<mvm_J[m_pyramid_level - level -1]<<std::endl<<std::endl;
      // std::cout<<mvm_inv_hessian[m_pyramid_level - level -1]<<std::endl<<std::endl;
      delta_x *= 2.f; // correct x location for going to lower scale
      ProcessInLayer_stereoX(delta_x, mm_H0s[m_pyramid_level - level -1], error, max_iter, epsilon, winsize,
                      vm_current_image_pyramid[m_pyramid_level - level - 1], // current image to track new location
                      mvm_ref_image_pyramid[m_pyramid_level - level - 1], // Reference image window to allign with
                      m_pyramid_level - level - 1, // curent pyramid level analyzed
                      mvm_inv_hessian[m_pyramid_level - level -1], // pre computed inverse hessian matrix
                      mvm_J[m_pyramid_level - level -1]); // pre computed image gradient of reference image window
    }
    curr_points[pt].x += delta_x;
    inlier_markers.push_back(error < error_threshold);
  }
  return ;
}


bool WarpCurrentImage_stereoX(const cv::Mat& src, cv::Mat &dst, const float& dx_i,
                        const cv::Point2f& x0y0, const cv::Size& winsize)
{
  static int x;
  static int y;
  static float x_sub;
  static float y_sub;
  static cv::Mat1f interpolate_img = cv::Mat::zeros(winsize.height, winsize.width, CV_32F);
  static cv::Rect warp_interpolated_img = cv::Rect(cv::Point(0, 0), interpolate_img.size());

  //check if the warp is inside the border of the src image
  if ( x0y0.x + dx_i >= 0 &&
        x0y0.x + dx_i + 1  + winsize.width <  src.cols && // +1 for interpolation and/or
        x0y0.y >= 0 &&       
        x0y0.y + 1 + winsize.height <  src.rows) // rounding to nearest neigbour
  {
    x = x0y0.x + dx_i;
    y = x0y0.y;
    x_sub = x0y0.x + dx_i - x;
    y_sub = x0y0.y - y;

    static float s1, s2, s3, s4;
    s1 = (1.f-x_sub)*(1.f-y_sub);
    s2 = (1.f-x_sub)*y_sub;
    s3 = (1.f-y_sub)*x_sub;
    s4 = x_sub*y_sub;
    // if (dst.isContinuous() && src.isContinuous())
    // {
      static int warpCols = winsize.width;
      static int imCols;
      imCols = src.cols;
      static int i,j;
      static float* p;
      static const uchar* q;
      p = dst.ptr<float>(0);
      q = src.ptr<uchar>(y);
      for( i = 0; i <  winsize.height; ++i)
      {
          for ( j = 0; j < warpCols; ++j)
          {
            p[j + i*warpCols] = s1*q[x + j + i*imCols]
                            + s2*q[x + j + (i+1)*imCols]
                            + s3*q[x + j+1 + i*imCols]
                            + s4*q[x + j+1 + (i+1)*imCols];
          }
      }
    // }
    // else
    // {
    //   std::cout<<"ERROR::WarpCurrentImage_stereoX:: cv mat not aligned"<<std::endl;
    // }
  }
  else return false;
  return true;
}

//@todo: make warp interpolater a void, create interpolate img matrix outside this function and use them as input/ouput 
cv::Mat warp_interpolater_stereoX(const cv::Mat& src, const float& x0, const float& y0, const cv::Size& winsize)
{
  cv::Mat1f interpolate_img = cv::Mat::zeros(winsize.height+1, winsize.width+1, CV_32F); // maybe width andheight reversed
  static float x_sub, y_sub;
  static int x, y;
  static float s1, s2, s3, s4;
  x = int(x0);
  y = int(y0);

  x_sub = x0 - x;
  y_sub = y0 - y;
  s1 = (1.f-x_sub)*(1.f-y_sub);
  s2 = (1.f-x_sub)*y_sub;
  s3 = (1.f-y_sub)*x_sub;
  s4 = x_sub*y_sub;

  static int warpCols = winsize.width+1;
  static int imCols;
  imCols = src.cols;

  static int i,j;
  static float* p;
  static const uchar* q;
  p = interpolate_img.ptr<float>(0);
  q = src.ptr<uchar>(y);
  // if (interpolate_img.isContinuous() && src.isContinuous())
  // {
    for( i = 0; i <  winsize.height+1; ++i)
    {
        for ( j = 0; j < warpCols; ++j)
        {
          // std::cout<<i<<j<<std::endl;
          p[j + i*warpCols] = s1*q[x + j + i*imCols]
                          + s2*q[x + j + (i+1)*imCols]
                          + s3*q[x + j+1 + i*imCols]
                          + s4*q[x + j+1 + (i+1)*imCols];
        }
    }
  // }
  // else
  // {
  //   std::cout<<"WARNING::warp_interpolater_stereoX:: cv mat not aligned, this decrease computational performance"<<std::endl;
  //   for(i=0; i<winsize.width+1; i++)
  //   {
  //     for(j=0; j<winsize.height+1; j++)
  //     {
  //       interpolate_img.at<float>(j,i) = (float)(((1.f-x_sub)*(1.f-y_sub))*(src.at<uchar>(y+j,x+i)) // put x0 in i and y0 in j
  //                           + (1.f-x_sub)*y_sub*(src.at<uchar>(y+1+j, x+i))
  //                           + (1.f-y_sub)*x_sub*(src.at<uchar>(y+j, x+1+i))
  //                           + x_sub*y_sub*(src.at<uchar>(y+1+j, x+1+i)));
  //     }
  //   }
  // }
  return interpolate_img;
}
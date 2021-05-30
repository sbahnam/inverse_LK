#include "LK20_class.hpp"

invLK::invLK(const int pyr_size, const float pyr_scale, const cv::Size img_size, cv::Size win_size)
{
  this->winsize = win_size;
  this->m_pyramid_level = pyr_size;
  this->scale_level = pyr_scale;

  for (int i=0; i<pyr_size; i++)
  {
    this->mvm_J.push_back(cv::Mat::zeros(2, win_size.width*win_size.height, CV_32F));
    this->mvm_inv_hessian.push_back(cv::Mat::zeros(2, 2, CV_32F));
    this->mm_H0s.push_back(cv::Point2f(0.f, 0.f));
    this->mvm_ref_image_pyramid.push_back(cv::Mat::zeros(win_size.width+2, win_size.height+2, CV_32F));
  }
}


void invLK::ComputeImageGradient(const int& l)
{
  static int imCols = this->mvm_ref_image_pyramid[l].cols;
  static int i,j;
  static float* dx;
  static float* dy;
  static const float* qup;
  static const float* ql;
  static const float* qr;
  static const float* qbot;
  static int k;

  dx = this->mvm_J[l].ptr<float>(0);
  dy = this->mvm_J[l].ptr<float>(1,0);
  k = 0;

  for( i = 0; i < this->mvm_ref_image_pyramid[l].rows-2; ++i)
  {
    qup = this->mvm_ref_image_pyramid[l].ptr<float>(i,1);     // | Aij   Aqup    .. |
    ql = qup + imCols - 1;                                    // | Aql  Ai+1j+1  Aqr|
    qr = ql + 2;                                              // |Ai+2j   Abot   .. |
    qbot = qr + imCols - 1;                                   // we calculate the gradient around Ai+1j+1
    for ( j = 0; j < this->mvm_ref_image_pyramid[l].rows-2; ++j)
      {
        dx[k] = qr[j] - ql[j];
        dy[k] = qbot[j] - qup[j];
        k++;
      }
  }

  return;
}


void invLK::ComputeInvHessian(const int& l) {
  static float D;
  static float* p;
  static const float* dx;
  static const float* dy;
  static int i;

  p = this->mvm_inv_hessian[l].ptr<float>(0); // @why? we do not need to reset the indices to zero
                                              // because it is already hapenning without specifying it
  dx = this->mvm_J[l].ptr<float>(0);
  dy = this->mvm_J[l].ptr<float>(1,0);

  for( i = 0; i <  this->mvm_J[l].cols; ++i)
  {
    p[3] += dx[i] * dx[i]; // Acutally H00, where H is the hessian matrix
    p[1] -= dx[i] * dy[i]; // Actually -H01
    p[0] += dy[i] * dy[i]; // Actually H11
  }

  D = p[3]*p[0] - p[1]*p[1]; // H01=H10
  p[0] /= D;
  p[1] /= D;
  p[2] = p[1];
  p[3] /= D;

  return;
}


void invLK::ComputeResiduals(cv::Mat& m_residuals, const cv::Mat& cur_image, const int& level)
{
  static float f_residual;
  static float* p;
  static const float* q;
  static const float* r;
  static int i, j, k;

  f_residual = 0.f;
  p = m_residuals.ptr<float>(0);
  q = cur_image.ptr<float>(0);
  k = 0;

  for( i = 0; i < cur_image.rows; ++i)
  {
    r = this->mvm_ref_image_pyramid[level].ptr<float>(i+1,1);
    for (j = 0; j < cur_image.cols; ++j)
    {
      p[k] = q[k] - r[j];
      f_residual += p[k]*p[k];
      k++;
    }
  }

  this->sq_error = f_residual/(cur_image.rows*cur_image.cols);

  return;
}


void invLK::ComputeUpdateParams(const cv::Mat& m_residuals, const int& l, cv::Point2f& update)
{
  static float dp1;
  static float dp2;
  static const float* dx;
  static const float* dy;
  static const float* r;
  static int i;

  dp1 = 0;
  dp2 = 0;
  dx = this->mvm_J[l].ptr<float>(0);
  dy = this->mvm_J[l].ptr<float>(1,0);
  r = m_residuals.ptr<float>(0);

  for(i = 0; i < m_residuals.rows * m_residuals.cols; ++i)
  {
    dp1 += dx[i]*r[i];
    dp2 += dy[i]*r[i];
  }

  // 2x2 matrix calculation:
  update.x = this->mvm_inv_hessian[l].at<float>(0,0) * dp1 +
              this->mvm_inv_hessian[l].at<float>(0,1) * dp2;
  update.y = this->mvm_inv_hessian[l].at<float>(1,0) * dp1 +
              this->mvm_inv_hessian[l].at<float>(1,1) * dp2;

  return;
}


void invLK::ProcessInLayer(const int& max_iter,
                            const float& epsilon,
                            const cv::Mat& m_cur_image_in_pyramid,
                            const int& level)
{
  static cv::Mat1f m_residuals = cv::Mat::zeros(this->winsize.height*this->winsize.width, 1, CV_32F);
  static cv::Mat1f m_cur_working_image = cv::Mat::zeros(this->winsize.height, this->winsize.width, CV_32F);
  static float dx_i;
  static float dy_i;
  static cv::Point2f dxdxy;
  static cv::Point2f update;
  static bool success_warp;
  static int argmin;
  static std::vector<float> sq_errors_lst;
  static std::vector<cv::Point2f> delta_xys;

  sq_errors_lst.clear();
  delta_xys.clear();
  dxdxy = this->delta;

  for(int itr = 0; itr < max_iter; itr++)
  {
    success_warp = WarpCurrentImage(m_cur_image_in_pyramid, m_cur_working_image, dxdxy.x, dxdxy.y, this->mm_H0s[level]);
    if (success_warp != true) break; // This happens if part of the window goes outside of the image (no border used)

    ComputeResiduals(m_residuals, m_cur_working_image, level);
    sq_errors_lst.push_back(this->sq_error);
    delta_xys.push_back(dxdxy);

    // fprintf(stderr,"Itr[%d|%d]: %.6f\n",itr,level,this->sq_error);

    ComputeUpdateParams(m_residuals, level, update);

    dxdxy.x -= update.x;
    dxdxy.y -= update.y;
    if(update.x*update.x + update.y*update.y < epsilon*epsilon) break;
  }

  if (sq_errors_lst.size()>0) // check if first window is not already outside the image
  {
    argmin = std::min_element(sq_errors_lst.begin(), sq_errors_lst.end()) - sq_errors_lst.begin();
    this->delta = delta_xys[argmin];
    this->sq_error = sq_errors_lst[argmin];
  }

  return;
}


void invLK::TrackLKInv(const std::vector<cv::Mat>& vm_image_pyramid, const std::vector<cv::Mat>& vm_current_image_pyramid,
                      const std::vector<cv::Point2f>& prev_points, std::vector<cv::Point2f> &curr_points,
                      std::vector<unsigned char> &inlier_markers, const float sq_error_threshold, const int max_iter,
                      const float epsilon, const float max_distance)
{
  static int valid_levels;
  static float scaled;
  static float float_prev_rect_scaled_x;
  static float float_prev_rect_scaled_y;

  for (int pt=0; pt<prev_points.size(); pt++)
  {
    valid_levels = this->m_pyramid_level;
    this->sq_error = 9999.f; // should be initialized higher than sq_error_threshold
                             // in case the window is outside the image for all scales
    for (int l = 0; l < this->m_pyramid_level; l++)
    {
      if (l==0)
      {
        float_prev_rect_scaled_x = prev_points[pt].x - (this->winsize.width - 1)/2.f;
        float_prev_rect_scaled_y =prev_points[pt].y - (this->winsize.height - 1)/2.f;
      }
      else
      {
        scaled = vm_image_pyramid[0].cols/vm_image_pyramid[l].cols;
        float_prev_rect_scaled_x = (prev_points[pt].x - (scaled-1.f)/2.f)/scaled - (this->winsize.width-1)/2.f;
        float_prev_rect_scaled_y = (prev_points[pt].y - (scaled-1.f)/2.f)/scaled - (this->winsize.height-1)/2.f;
      }

      // @todo: check curr as well here,so you can discard some of them and do not require to do that later
      if (float_prev_rect_scaled_x >= -1 && float_prev_rect_scaled_y >= -1 && // -1 to calculate gradient at border
            float_prev_rect_scaled_x + this->winsize.width + 2 < vm_image_pyramid[l].cols && // +2 because image gradient uses
            float_prev_rect_scaled_y + this->winsize.height + 2 < vm_image_pyramid[l].rows)  // +1 to calculate values at border
                                                                                             // and +1 for interpolation
                                                                                              // @todo: it should be +1!?
      {
        warp_interpolater(vm_image_pyramid[l], l, float_prev_rect_scaled_x, float_prev_rect_scaled_y);
      }
      else
      {
        valid_levels = l;
        break;
      }

      if (l==0)
        this->mm_H0s[l] = cv::Point2f(curr_points[pt].x-(this->winsize.width - 1)/2.f, 
                        curr_points[pt].y-(this->winsize.height - 1)/2.f);
      else
      {
        this->mm_H0s[l] = cv::Point2f((curr_points[pt].x - (scaled-1.f)/2.f)/scaled - (this->winsize.width -1)/2.f,
                                  (curr_points[pt].y - (scaled-1.f)/2.f)/scaled - (this->winsize.height -1)/2.f);
      }

      // Pre compute gradient and hessian on each pyramid (inverse compositional method)
      ComputeImageGradient(l);
      ComputeInvHessian(l);
    }

    this->delta.x=0.f;
    this->delta.y=0.f;

    // Coarse-to-Fine Optimization 
    for (int level = valid_levels-1; level >= 0; level--)
    {
      this->delta.x *= this->scale_level; // correct x and y location for going from
      this->delta.y *= this->scale_level; // higher scale to lower/orignal scale
      ProcessInLayer(max_iter, epsilon, vm_current_image_pyramid[level], level);
    }

    curr_points[pt].x += this->delta.x;
    curr_points[pt].y += this->delta.y;
    inlier_markers.push_back(this->sq_error < sq_error_threshold &&  this->delta.x * this->delta.x + this->delta.y *this->delta.y < max_distance);
  }

  return ;
}


bool invLK::WarpCurrentImage(const cv::Mat& src, cv::Mat &dst, const float& dx_i,
                              const float& dy_i, const cv::Point2f& x0y0)
{
  static int x;
  static int y;
  static float x_sub;
  static float y_sub;
  static bool success;

  //check if the warp is inside the border of the src image
  if ( x0y0.x + dx_i >= 0 &&
        x0y0.x + dx_i + 1  + this->winsize.width <  src.cols && // +1 for interpolation and/or
        x0y0.y + dy_i >= 0 &&       
        x0y0.y +  dy_i + 1 + this->winsize.height <  src.rows) // rounding to nearest neigbour
  {
    static float s1, s2, s3, s4;
    static int warpCols = this->winsize.width;
    static int i, j, k;
    static int imCols;
    static float* p;
    static const uchar* q1;
    static const uchar* q2;
    static const uchar* q3;
    static const uchar* q4;

    x = x0y0.x + dx_i;
    y = x0y0.y + dy_i;
    x_sub = x0y0.x + dx_i - x;
    y_sub = x0y0.y + dy_i - y;

    s1 = (1.f-x_sub)*(1.f-y_sub);
    s2 = (1.f-x_sub)*y_sub;
    s3 = (1.f-y_sub)*x_sub;
    s4 = x_sub*y_sub;

    imCols = src.cols;
    p = dst.ptr<float>(0);
    k = 0;

    for( i = 0; i <  this->winsize.height; ++i)
    {
      q1 = src.ptr<uchar>(y+i, x);    // | .. ..  ..  ..|   Where Aq1 = Aij
      q2 = q1 + imCols;               // | .. Aq1 Aq3 ..|
      q3 = q1 + 1;                    // | .. Aq2 Aq4 ..|
      q4 = q2 + 1;                    // | .. ..  ..  ..|
      for ( j = 0; j < warpCols; ++j)
      {
        p[k] = s1*q1[j] + s2*q2[j] + s3*q3[j] + s4*q4[j];
        k++;
      }
    }
  success = true;
  }
  else success = false;

  return success;
}


void invLK::warp_interpolater(const cv::Mat& src, int& l, const float& x0, const float& y0)
{
  static float x_sub, y_sub;
  static int x, y;
  static float s1, s2, s3, s4;
  static int warpCols = this->winsize.width+2;
  static int warpRows = this->winsize.height+2;
  static int imCols;
  static int i, j, k;
  static float* p;
  static const uchar* q1;
  static const uchar* q2;
  static const uchar* q3;
  static const uchar* q4;

  x = int(x0);
  y = int(y0);
  x_sub = x0 - x;
  y_sub = y0 - y;

  s1 = (1.f-x_sub)*(1.f-y_sub);
  s2 = (1.f-x_sub)*y_sub;
  s3 = (1.f-y_sub)*x_sub;
  s4 = x_sub*y_sub;

  imCols = src.cols;
  p = this->mvm_ref_image_pyramid[l].ptr<float>(0);
  k = 0;

  for( i = 0; i <  warpRows; ++i)
  {
    q1 = src.ptr<uchar>(y-1+i, x-1);  // | .. ..  ..  ..|   Where Aq1 = A(i-1)(j-1)
    q2 = q1 + imCols;                 // | .. Aq1 Aq3 ..|   -1 needed for gradient calculation
    q3 = q1 + 1;                      // | .. Aq2 Aq4 ..|
    q4 = q2 + 1;                      // | .. ..  ..  ..|
    for ( j = 0; j < warpCols; ++j)
    {
      p[k] = s1*q1[j] + s2*q2[j] + s3*q3[j] + s4*q4[j];
      k++;
    }
  }

  return;
}
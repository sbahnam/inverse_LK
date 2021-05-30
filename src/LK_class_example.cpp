#include <iostream>
#include <chrono>
#include "LK_class_example.hpp"
#include "LK20_class.hpp"


void GenerateImagePyramid(const cv::Mat& m_src, std::vector<cv::Mat>& dst_pyramid) {
  if(m_src.channels() != 1) {
    std::cout<<"WARNING: A non grayscale image is givedst_pyramidn as input, grayscale is expected!"<<std::endl;
  }
  dst_pyramid[0] = m_src;
  if (pyramid_levels==1) return;
  ScaleDown(m_src, dst_pyramid[1]);

  for(int l = 2; l < pyramid_levels; l++) {   
    ScaleDown(dst_pyramid[l-1], dst_pyramid[l]);
  }
}

void ScaleDown(const cv::Mat& src, cv::Mat& dst) {
  static int srcCols;
  srcCols = src.cols;

  static int i,j;
  // if(scaled_img.isContinuous())
  if(dst.isContinuous())
  {
    static int k;
    k =0;
    uchar* p;
    const uchar* q;
    p = dst.ptr<uchar>(0);
    if (pyramid_scale == 2)
    {
      for( i = 0; i < dst.rows; ++i)
      {
        q = src.ptr<uchar>(i+i);
          for ( j = 0; j < dst.cols; ++j)
          {
            p[k] = (q[j+j] + q[j+j + 1] +
              q[j+j + srcCols] +  q[j+j + 1 + srcCols])/4;
            k++;
          }
      }
    }
    else if (pyramid_scale == 3)
    {
      for( i = 0; i < dst.rows; ++i)
      {
        q = src.ptr<uchar>(3*i);
          for ( j = 0; j < dst.cols; ++j)
          {
            p[k] = (q[j*3] + q[j*3 + 1] + q[j*3 + 2] +
                    q[j*3 + srcCols] +  q[j*3 + 1 + srcCols] + q[j*3 + 2 + srcCols] +
                    q[j*3 + srcCols + srcCols] +  q[j*3 + 1 + srcCols + srcCols] + q[j*3 + 2 + srcCols + srcCols])/9;
            k++;
          }
      }
    }
    else if(pyramid_scale == 4)
    {
      for( i = 0; i < dst.rows; ++i)
      {
        q = src.ptr<uchar>(4*i);
          for ( j = 0; j < dst.cols; ++j)
          {
            // std::cout<<i<<", "<<j<<", "<<k<<std::endl;
            p[k] = (q[j*4] + q[j*4 + 1] + q[j*4 + 2] + q[j*4 + 3] +
                    q[j*4 + srcCols] +  q[j*4 + 1 + srcCols] + q[j*4 + 2 + srcCols] +  q[j*4 + 3 + srcCols] +
                    q[j*4 + srcCols + srcCols] +  q[j*4 + 1 + srcCols + srcCols] + q[j*4 + 2 + srcCols + srcCols] +  q[j*4 + 3 + srcCols + srcCols] +
                    q[j*4 + srcCols*3] +  q[j*4 + 1 + srcCols*3] + q[j*4 + 2 + srcCols*3] +  q[j*4 + 3 + srcCols*3])/16;
            k++;
          }
      }
    }
  }
  else
  {
    std::cout<<"WARNING::ImageProcessor::ScaleDown cv mat not continuous, increasing computation time"<<std::endl;
    for (i=0; i<dst.rows; i++)
    {
      for (j=0; j<dst.cols; j++)
      {
        dst.at<uchar>(i,j) = (src.at<uchar>(2*i,2*j) +
                                      src.at<uchar>(2*i+1,2*j) +
                                      src.at<uchar>(2*i,2*j+1) +
                                      src.at<uchar>(2*i+1,2*j+1) ) / 4;
      }
    }
  }
  return;
}

int main(int argc, char* argv[]){
  for (int level=0; level<pyramid_levels; level++)
    {
      static int scale = 1; 
      // int scale = 1+level; 
      // curr_cam1_pyramid_.push_back(cv::Mat::zeros(480/scale, 752/scale, CV_8U));
      prev_cam0_pyramid_.push_back(cv::Mat::zeros(480/scale, 752/scale, CV_8U));
      curr_cam0_pyramid_.push_back(cv::Mat::zeros(480/scale, 752/scale, CV_8U));
      scale *= pyramid_scale;
    }



  // cv::Mat prev_cam0_img = cv::imread("/home/stavrow/EuRoC_dataset/V1_01_easy/mav0/cam0/data/1403715273262142976.png", cv::IMREAD_GRAYSCALE);
  // cv::Mat curr_cam0_img = cv::imread("/home/stavrow/EuRoC_dataset/V1_01_easy/mav0/cam0/data/1403715273312143104.png", cv::IMREAD_GRAYSCALE);
  cv::Mat prev_cam0_img = cv::imread("/home/pi/EuRoC_dataset/EuRoC/V1_01_easy/mav0/cam0/data/1403715273262142976.png", cv::IMREAD_GRAYSCALE);
  cv::Mat curr_cam0_img = cv::imread("/home/pi/EuRoC_dataset/EuRoC/V1_01_easy/mav0/cam0/data/1403715273312143104.png", cv::IMREAD_GRAYSCALE);

  GenerateImagePyramid(prev_cam0_img, prev_cam0_pyramid_);
  GenerateImagePyramid(curr_cam0_img, curr_cam0_pyramid_);

  cv::Point2f p1 = cv::Point2f(246, 246);
  cv::Point2f p2 = cv::Point2f(380, 460);
  cv::Point2f p3 = cv::Point2f(120, 110);
  cv::Point2f p1_curr = cv::Point2f(241, 240);
  cv::Point2f p2_curr = cv::Point2f(383, 458);
  cv::Point2f p3_curr = cv::Point2f(125, 103);

  std::vector<cv::Point2f> prev_cam0_points;
  std::vector<cv::Point2f> curr_cam0_points;
  prev_cam0_points.push_back(p1);
  prev_cam0_points.push_back(p2);
  prev_cam0_points.push_back(p3);
  curr_cam0_points.push_back(p1_curr);
  curr_cam0_points.push_back(p2_curr);
  curr_cam0_points.push_back(p3_curr);
  std::vector<cv::Point2f> curr_cam0_points_restore = curr_cam0_points;

  std::vector<unsigned char> track_inliers(0);

  int img_width = curr_cam0_img.cols;
  int img_height = curr_cam0_img.rows;
  cv::Size winsize = cv::Size(15,15);
  float pyr_scale = 2.f;

  invLK tracker = invLK{curr_cam0_pyramid_.size(), pyr_scale, cv::Size(img_width,img_height), winsize};
  auto initialize = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = initialize - initialize;

  int itrs = 1000;
  for (int i=0; i<itrs; i++)
  {
    auto start = std::chrono::steady_clock::now();
    tracker.TrackLKInv(prev_cam0_pyramid_, curr_cam0_pyramid_, // Pyramid images
          prev_cam0_points, curr_cam0_points, // Previous feature location & estimated new location
          track_inliers, 30*30,
          30, 0.01, // Stop criteria
          9998); // max distance squared to be inlier
    auto end = std::chrono::steady_clock::now();
    elapsed_seconds += end-start;

    if (i < itrs - 1) curr_cam0_points = curr_cam0_points_restore;
  }

  for (int i=0; i<curr_cam0_points.size(); i++)
  {
    std::cout<<curr_cam0_points[i].x<<", "<<curr_cam0_points[i].y<<std::endl;
  }
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

  return 0;
}

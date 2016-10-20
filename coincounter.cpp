/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - load.cpp
// TOPIC: load and display an image
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;           //make available OpenCV namespace
using namespace std;

const double pi = atan(1)*4;

int abs(int x){
  if(x<0)
    return -x;
  return x;
}

int sqr(int x){
  return x*x;
}

void convolution3by3(const Mat& image, Mat& result, const Mat& kernel)
{
  result.create(image.size(),image.type());
  int weight = 0;
  for(int j=0;j<3;j++){
    for(int i=0;i<3;i++){
      weight += abs(kernel.at<char>(j,i));
    }
  }

  const char* k0 = kernel.ptr<char>(0);
  const char* k1 = kernel.ptr<char>(1);
  const char* k2 = kernel.ptr<char>(2);
  for(int j=1;j<image.rows-1;j++){
    const uchar* previous = image.ptr<uchar>(j-1);
    const uchar* current  = image.ptr<uchar>(j  );
    const uchar* next     = image.ptr<uchar>(j+1);
    for(int i=1;i<image.cols-1;i++){
      int res = previous[i-1]*k2[2] + previous[i]*k2[1] + previous[i+1]*k2[0]
                 + current[i-1]*k1[2] + current[i]*k1[1] + current[i+1]*k1[0]
                 + next[i-1]*k0[2] + next[i]*k0[1] + next[i+1]*k0[0];
      result.at<uchar>(j,i) = round(res/(double)weight+127.5);

    }
  }
  //cout << result << endl;
}

bool valid(int num, int min, int max){
  if(num<min)
    return false;
  if(num>max)
    return false;
  return true;
}

int hough(const Mat& original_thr, const Mat& grads, Mat &hspace, uchar hough_thr, uchar minR, uchar maxR){
  int sizes[3] = {original_thr.rows,original_thr.cols,maxR-minR};
  hspace.create(3,sizes,original_thr.type());
  for(int j=0;j<original_thr.rows;j++)
  {
    for(int i=0;i<original_thr.cols;i++)
    {
      for(int r=minR;r<maxR;r++)
      {
        if(original_thr.at<uchar>(j,i)==255)
        {
          int x0 = i+r*cos(grads.at<uchar>(j,i)/255.0*2*pi-pi);
          int y0 = j+r*sin(grads.at<uchar>(j,i)/255.0*2*pi-pi);
          if(valid(x0-1,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
            hspace.at<uchar>(y0-1,x0-1,r-minR) += 1;
          if(valid(x0,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
            hspace.at<uchar>(y0-1,x0,r-minR) += 1;
          if(valid(x0+1,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
            hspace.at<uchar>(y0-1,x0+1,r-minR) += 1;
          if(valid(x0-1,0,original_thr.cols) && valid(y0,0,original_thr.rows))
            hspace.at<uchar>(y0,x0-1,r-minR) += 1;
          if(valid(x0,0,original_thr.cols) && valid(y0,0,original_thr.rows))
            hspace.at<uchar>(y0,x0,r-minR) += 1;
          if(valid(x0+1,0,original_thr.cols) && valid(y0,0,original_thr.rows))
            hspace.at<uchar>(y0,x0+1,r-minR) += 1;
          if(valid(x0-1,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
            hspace.at<uchar>(y0+1,x0-1,r-minR) += 1;
          if(valid(x0,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
            hspace.at<uchar>(y0+1,x0,r-minR) += 1;
          if(valid(x0+1,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
            hspace.at<uchar>(y0+1,x0+1,r-minR) += 1;

          if(valid(r-1,minR,maxR))
          {
            if(valid(x0-1,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
              hspace.at<uchar>(y0-1,x0-1,r-minR-1) += 1;
            if(valid(x0,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
              hspace.at<uchar>(y0-1,x0,r-minR-1) += 1;
            if(valid(x0+1,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
              hspace.at<uchar>(y0-1,x0+1,r-minR-1) += 1;
            if(valid(x0-1,0,original_thr.cols) && valid(y0,0,original_thr.rows))
              hspace.at<uchar>(y0,x0-1,r-minR-1) += 1;
            if(valid(x0,0,original_thr.cols) && valid(y0,0,original_thr.rows))
              hspace.at<uchar>(y0,x0,r-minR-1) += 1;
            if(valid(x0+1,0,original_thr.cols) && valid(y0,0,original_thr.rows))
              hspace.at<uchar>(y0,x0+1,r-minR-1) += 1;
            if(valid(x0-1,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
              hspace.at<uchar>(y0+1,x0-1,r-minR-1) += 1;
            if(valid(x0,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
              hspace.at<uchar>(y0+1,x0,r-minR-1) += 1;
            if(valid(x0+1,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
              hspace.at<uchar>(y0+1,x0+1,r-minR-1) += 1;
          }

          if(valid(r+1,minR,maxR))
          {
            if(valid(x0-1,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
              hspace.at<uchar>(y0-1,x0-1,r-minR) += 1;
            if(valid(x0,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
              hspace.at<uchar>(y0-1,x0,r-minR) += 1;
            if(valid(x0+1,0,original_thr.cols) && valid(y0-1,0,original_thr.rows))
              hspace.at<uchar>(y0-1,x0+1,r-minR) += 1;
            if(valid(x0-1,0,original_thr.cols) && valid(y0,0,original_thr.rows))
              hspace.at<uchar>(y0,x0-1,r-minR) += 1;
            if(valid(x0,0,original_thr.cols) && valid(y0,0,original_thr.rows))
              hspace.at<uchar>(y0,x0,r-minR) += 1;
            if(valid(x0+1,0,original_thr.cols) && valid(y0,0,original_thr.rows))
              hspace.at<uchar>(y0,x0+1,r-minR) += 1;
            if(valid(x0-1,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
              hspace.at<uchar>(y0+1,x0-1,r-minR) += 1;
            if(valid(x0,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
              hspace.at<uchar>(y0+1,x0,r-minR) += 1;
            if(valid(x0+1,0,original_thr.cols) && valid(y0+1,0,original_thr.rows))
              hspace.at<uchar>(y0+1,x0+1,r-minR) += 1;
          }
        }
      }
    }
  }
  Mat hspace2D;
  hspace2D.create(original_thr.size(),original_thr.type());
  Mat hspace2DLog;
  hspace2DLog.create(hspace2D.size(),hspace2D.type());
  int count = 0;
  for(int j=0;j<original_thr.rows;j++)
  {
    for(int i=0;i<original_thr.cols;i++)
    {
      hspace2D.at<uchar>(j,i) = 0;
      for(int r=minR;r<maxR;r++)
      {
        hspace2D.at<uchar>(j,i) += hspace.at<uchar>(j,i,r-minR);
        if(hspace.at<uchar>(j,i,r-minR)>hough_thr){
          cout << "Coin found: (" << i <<","<<j<<") r="<<r<<endl;
          count++;
        }
      }
      hspace2DLog.at<uchar>(j,i)=50*log(1+hspace2D.at<uchar>(j,i));
    }
  }
  namedWindow("Hough 2D", CV_WINDOW_AUTOSIZE);
  imshow("Hough 2D", hspace2DLog);
  waitKey(0);

  return count;
}

void sobel(const Mat& image, Mat& dx, Mat& dy, Mat& mag, Mat& dir){
  Mat dxKern = (Mat_<char>(3,3) << -1,  0,  1,
                                   -2,  0,  2,
                                   -1,  0,  1);
  Mat dyKern = (Mat_<char>(3,3) << -1, -2, -1,
                                    0,  0,  0,
                                    1,  2,  1);

  convolution3by3(image,dx,dxKern);
  convolution3by3(image,dy,dyKern);

  mag.create(image.size(),image.type());
  for(int j=1;j<image.rows-1;j++)
  {
    for(int i=1;i<image.cols-1;i++){
      mag.at<uchar>(j,i) = sqrt(sqr(dx.at<uchar>(j,i)-127)+sqr(dy.at<uchar>(j,i)-127));
    }
  }

  dir.create(image.size(),image.type());
  for(int j=1;j<image.rows-1;j++)
  {
    for(int i=1;i<image.cols-1;i++){
      double r = atan2((double)(dy.at<uchar>(j,i)-127),(double)(dx.at<uchar>(j,i)-127));
      dir.at<uchar>(j,i) = (uchar)255*((r-(-pi))/(pi-(-pi))); //scale
    }
  }
  //cout << dir << endl;

  namedWindow("Dy sobel", CV_WINDOW_AUTOSIZE);
  imshow("Dy sobel", dy);
  waitKey(0);


  namedWindow("Dx sobel", CV_WINDOW_AUTOSIZE);
  imshow("Dx sobel", dx);
  waitKey(0);

  namedWindow("Mag sobel", CV_WINDOW_AUTOSIZE);
  imshow("Mag sobel", mag);
  waitKey(0);

  namedWindow("Dir sobel", CV_WINDOW_AUTOSIZE);
  imshow("Dir sobel", dir);
  waitKey(0);

}

void threshold(const Mat& original, Mat& result){
  result.create(original.size(),original.type());
  for(int j=0;j<original.rows;j++)
  {
    for(int i=0;i<original.cols;i++)
    {
      if(original.at<uchar>(j,i)>30)
      {
        result.at<uchar>(j,i)=255;
      }
      else
      {
        result.at<uchar>(j,i)=0;
      }
    }
  }
  namedWindow("Threshold sobel magnitude", CV_WINDOW_AUTOSIZE);
  imshow("Threshold sobel magnitude", result);
  waitKey(0);
}


int main() {

  //declare a matrix container to hold an image
  Mat image;
  Mat dx,dy,mag,dir,thres,grads,hspace;

  //load image from a file into the container
  image = imread("coins1.png", IMREAD_GRAYSCALE);
  //imwrite("gray.png",image);

  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  imshow("Display window", image);

  //wait for a key press until returning from the program
  waitKey(0);

  sobel(image,dx,dy,mag,dir);

  threshold(mag, thres);

  int count = hough(thres, dir, hspace, 50, 26, 30);
  cout << "Coins detected: " << count << endl;
  //free memory occupied by image
  image.release();

  return 0;
}

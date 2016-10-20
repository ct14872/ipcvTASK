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
  for(int j=0;j<image.rows;j++)
  {
    for(int i=0;i<image.cols;i++){
      mag.at<uchar>(j,i) = sqrt(sqr(dx.at<uchar>(j,i)-127)+sqr(dy.at<uchar>(j,i)-127));
    }
  }

  dir.create(image.size(),image.type());
  for(int j=0;j<image.rows;j++)
  {
    for(int i=0;i<image.cols;i++){
      dir.at<uchar>(j,i) = atan((double)(dy.at<uchar>(j,i)-127)/(double)(dx.at<uchar>(j,i)-127));
    }
  }

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


int main() {

  //declare a matrix container to hold an image
  Mat image;
  Mat dx,dy,mag,dir;

  //load image from a file into the container
  image = imread("coins2.png", IMREAD_GRAYSCALE);
  //imwrite("gray.png",image);

  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  imshow("Display window", image);

  //wait for a key press until returning from the program
  waitKey(0);

  sobel(image,dx,dy,mag,dir);

  //free memory occupied by image
  image.release();

  return 0;
}

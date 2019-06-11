#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>

using namespace cv;
using namespace std;

#define CONFIG_MASK_SIZE 32
#define CONFIG_IMAGE_WIDTH 640
#define CONFIG_IMAGE_HEIGHT 480
#define CONFIG_IMAGE_CHANNEL 3

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}


extern
void LaunchKernel(
        const unsigned char *srcImage,
        const unsigned char *mask,
        unsigned char *dstImage);

int mainold(int argc, char** argv)
{
    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return 0;
    for(;;)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() ) break; // end of video stream
        imshow("this is you, smile! :)", frame);
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}

int main(int argc, char** argv)
{
    const unsigned long lenImage = CONFIG_IMAGE_HEIGHT * CONFIG_IMAGE_WIDTH * 3;
    const unsigned long lenMask  = CONFIG_MASK_SIZE * CONFIG_MASK_SIZE;

    unsigned char *h_src1;
    unsigned char *d_src1;

    unsigned char *h_dst1;
    unsigned char *d_dst1;

    unsigned char *h_mask;
    unsigned char *d_mask;


    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    cout<<"Selected CUDA Device: "<< dev<<", "<< deviceProp.name<<endl;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaDeviceReset());
    CHECK(cudaDeviceSynchronize());


    Mat imgMask = imread("../mask.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    if(!imgMask.data){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    h_src1 = (unsigned char*)malloc(sizeof(unsigned char) * lenImage);
    h_dst1 = (unsigned char*)malloc(sizeof(unsigned char) * lenImage);
    h_mask = (unsigned char*)malloc(sizeof(unsigned char) * lenMask);

    for(unsigned long i=0; i<lenMask; i++){
        h_mask[i] = imgMask.data[i];
    }

    CHECK(cudaMalloc((void**)&d_src1, sizeof(unsigned char) * lenImage));
    CHECK(cudaMalloc((void**)&d_mask, sizeof(unsigned char) * lenMask));
    CHECK(cudaMalloc((void**)&d_dst1, sizeof(unsigned char) * lenImage));

    // Transfer data from host to device memory
    CHECK(cudaMemcpy(d_mask, h_mask, sizeof(unsigned char) * lenMask, cudaMemcpyHostToDevice));




    VideoCapture cap;
    if(!cap.open(0))
        return 0;

    int cnt=0;
    //for(int iframe=0; iframe<2; iframe++) {
    for(;;){
        Mat frame;
        cap >> frame;
        if( frame.empty() ) break; // end of video stream
        //=======================================================================================
        for(unsigned long i = 0 ; i<lenImage;i++){
            h_src1[i] = frame.data[i];
        }


        CHECK(cudaMemcpy(d_src1, h_src1, sizeof(unsigned char) * lenImage, cudaMemcpyHostToDevice));
        LaunchKernel(
                d_src1,
                d_mask,
                d_dst1);
        CHECK(cudaMemcpy(h_dst1, d_dst1, sizeof(unsigned char) * lenImage, cudaMemcpyDeviceToHost));

        cv::Mat img(CONFIG_IMAGE_HEIGHT, CONFIG_IMAGE_WIDTH, CV_8UC3, h_dst1);
        imshow("Web Camera Source", frame);
        imshow("GPU Output", img);
        if(waitKey(1)==27)break;
        cout<<"frame " << cnt++ << endl;
    }


    cudaFree(d_src1);
    cudaFree(d_mask);
    cudaFree(d_dst1);
    free(h_src1);
    free(h_mask);
    free(h_dst1);
}
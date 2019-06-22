#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>

using namespace cv;
using namespace std;


#define CONFIG_MASK_SIZE  3
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
void LaunchKernel_SpatialFilter(
        const unsigned char *srcImage,
        const float *mask,
        unsigned char *dstImage);

int main(int argc, char** argv)
{
    const unsigned long lenImage = CONFIG_IMAGE_HEIGHT * CONFIG_IMAGE_WIDTH * 3;
    const unsigned long lenMask  = CONFIG_MASK_SIZE * CONFIG_MASK_SIZE;

    unsigned char *d_src1;

    unsigned char *h_dst1;
    unsigned char *d_dst1;

    float h_mask[] = {-1,0,1,-2,0,2,-1,0,1};
    float *d_mask;


    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    cout<<"Selected CUDA Device: "<< dev<<", "<< deviceProp.name<<endl;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaDeviceReset());
    CHECK(cudaDeviceSynchronize());


    h_dst1 = (unsigned char*)malloc(sizeof(unsigned char) * lenImage);



    CHECK(cudaMalloc((void**)&d_src1, sizeof(unsigned char) * lenImage));
    CHECK(cudaMalloc((void**)&d_mask, sizeof(float) * lenMask));
    CHECK(cudaMalloc((void**)&d_dst1, sizeof(unsigned char) * lenImage));

    // Transfer data from host to device memory
    CHECK(cudaMemcpy(d_mask, h_mask, sizeof(float) * lenMask, cudaMemcpyHostToDevice));




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

        CHECK(cudaMemcpy(d_src1, frame.data, sizeof(unsigned char) * lenImage, cudaMemcpyHostToDevice));
        LaunchKernel_SpatialFilter(
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
    free(h_dst1);
}
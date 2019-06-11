#include<cuda.h>
#include<cuda_runtime.h>
#include <stdio.h>

#define CONFIG_MASK_SIZE 32
#define CONFIG_IMAGE_WIDTH 640
#define CONFIG_IMAGE_HEIGHT 480
#define CONFIG_IMAGE_CHANNEL 3

template <int BLOCK_SIZE, int MASK_SIZE, int CHANNELS>
__global__ void WatermarkKernel(
        const unsigned char *srcImage, //HxWxC
        const unsigned char *mask, // CONFIG_MASK_SIZE x CONFIG_MASK_SIZE
        unsigned char *dstImage){

    __shared__ float localMem[MASK_SIZE][MASK_SIZE][CHANNELS];
    unsigned long index, maskIndex;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Global index
    int gx = bx*BLOCK_SIZE+tx;
    int gy = by*BLOCK_SIZE+ty;


    index = gy*CONFIG_IMAGE_WIDTH*CONFIG_IMAGE_CHANNEL + gx*CONFIG_IMAGE_CHANNEL;
    for(int c=0; c<3; c++){
        maskIndex = ty*MASK_SIZE + tx;
        localMem[ty][tx][c] = (unsigned char) (srcImage[index+c] * (float)mask[maskIndex] / 256.0f);
    }

    __syncthreads();

    for(int c=0; c<3; c++){
        localMem[ty][tx][c] = localMem[tx][ty][c];
    }

    __syncthreads();

    for(int c=0; c<3; c++){
        dstImage[index+c] = (unsigned char) (localMem[ty][tx][c]);
    }
}

void LaunchKernel(
        const unsigned char *srcImage,
        const unsigned char *mask,
        unsigned char *dstImage){

    dim3 block(CONFIG_MASK_SIZE,CONFIG_MASK_SIZE,1);
    dim3 grid(
            CONFIG_IMAGE_WIDTH/CONFIG_MASK_SIZE,
            CONFIG_IMAGE_HEIGHT/CONFIG_MASK_SIZE,
            1);

    //printf("BLOCK = (%d,%d,%d)\n", block.x, block.y, block.z);
    //printf("GRID  = (%d,%d,%d)\n", grid.x, grid.y, grid.z);
    WatermarkKernel<CONFIG_MASK_SIZE, CONFIG_MASK_SIZE, CONFIG_IMAGE_CHANNEL> <<<grid, block>>>(
            srcImage,
            mask,
            dstImage);
}
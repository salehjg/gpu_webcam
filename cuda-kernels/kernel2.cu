#include<cuda.h>
#include<cuda_runtime.h>
#include <stdio.h>

#define Mask_width  3
#define Mask_width_half  (Mask_width/2)

//Tiles are smaller than blocks, so we can pad the input image while burst reading it into local memory.

#define BLOCK_WIDTH 16
#define TILE_WIDTH (BLOCK_WIDTH - (Mask_width -1))


__global__ void SpatialFilter(
        const unsigned char* deviceInputImageData,
        unsigned char* deviceOutputImageData,
        const float* deviceMaskData,
        int imageWidth,
        int imageHeight,
        int imageChannels){
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];

    int row_o = ty + (blockIdx.y*TILE_WIDTH);
    int col_o = tx + (blockIdx.x*TILE_WIDTH);

    int row_i = row_o - Mask_width_half;
    int col_i = col_o - Mask_width_half;

    for(int color=0; color<imageChannels; color++){

        if( (row_i>=0) && (col_i>=0) && (row_i<imageHeight) && (col_i<imageWidth))
        {
            Ns[ty][tx] = deviceInputImageData[(row_i*imageWidth+col_i)*3+color];
        }
        else
        {
            Ns[ty][tx] = 0.0f;
        }

        __syncthreads();



        float output = 0.0f;

        int i,j;
        if( ty< TILE_WIDTH && tx<TILE_WIDTH)
        {
            for(i=0;i<Mask_width;i++)
            {
                for(j=0;j<Mask_width;j++)
                {
                    output = output + Ns[ty+i][tx+j]*deviceMaskData[i*Mask_width+j];

                }
            }
        }
        __syncthreads();

        if(tx < TILE_WIDTH && ty <TILE_WIDTH && row_o < imageHeight && col_o < imageWidth)
        {
            deviceOutputImageData[((row_o*imageWidth)+col_o)*3+color] = (unsigned char)output;
        }
    }
}


void LaunchKernel_SpatialFilter(
        const unsigned char *srcImage,
        const float *mask,
        unsigned char *dstImage){

    unsigned int imageWidth=640, imageHeight=480;

    dim3 block(BLOCK_WIDTH,BLOCK_WIDTH);
    dim3 grid((imageWidth-1)/TILE_WIDTH+1,(imageHeight-1)/TILE_WIDTH+1,1);

    SpatialFilter<<<grid,block>>>(srcImage, dstImage, mask, 640, 480, 3);
}
#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height, int channels);

__global__ void rgbToGrayKernel(unsigned char* input, unsigned char* output,
                                int width, int height, int channels);

__global__ void sobelKernel(unsigned char* input, unsigned char* output,
                           int width, int height);

#endif

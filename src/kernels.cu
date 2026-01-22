#include "kernels.cuh"
#include <math.h>

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return;

    float kernel[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    float sum = 273.0f;

    for (int c = 0; c < channels; c++) {
        float value = 0.0f;
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int idx = ((y + ky) * width + (x + kx)) * channels + c;
                value += input[idx] * kernel[ky + 2][kx + 2];
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)(value / sum);
    }
}

__global__ void rgbToGrayKernel(unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx >= total) return;

    if (channels == 1) {
        output[idx] = input[idx];
    } else {
        int rgbIdx = idx * channels;
        output[idx] = (unsigned char)(0.299f * input[rgbIdx + 2] + 
                                      0.587f * input[rgbIdx + 1] + 
                                      0.114f * input[rgbIdx]);
    }
}

__global__ void sobelKernel(unsigned char* input, unsigned char* output,
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int sumX = 0, sumY = 0;
    
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int idx = (y + ky) * width + (x + kx);
            sumX += input[idx] * Gx[ky + 1][kx + 1];
            sumY += input[idx] * Gy[ky + 1][kx + 1];
        }
    }
    
    int magnitude = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
    output[y * width + x] = (magnitude > 255) ? 255 : magnitude;
}

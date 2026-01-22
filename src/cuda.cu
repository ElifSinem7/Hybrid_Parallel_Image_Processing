#include "image_utils.h"
#include "kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input.bmp> <output.bmp>\n", argv[0]);
        return 1;
    }

    // Warm-up GPU
    cudaFree(0);

    double startTotal = getTime();
    
    // Load image
    double startIO = getTime();
    Image* img = loadBMP(argv[1]);
    if (!img) return 1;
    double endIO = getTime();

    printf("CUDA Implementation\n");
    printf("Image: %dx%d, %d channels\n", img->width, img->height, img->channels);
    printf("Load time: %.6f seconds\n", endIO - startIO);

    int size = img->width * img->height * img->channels;
    int graySize = img->width * img->height;

    // Allocate device memory
    unsigned char *d_input, *d_blurred, *d_gray, *d_edges;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_blurred, size));
    CUDA_CHECK(cudaMalloc(&d_gray, graySize));
    CUDA_CHECK(cudaMalloc(&d_edges, graySize));

    // Copy input to device
    double startTransfer = getTime();
    CUDA_CHECK(cudaMemcpy(d_input, img->data, size, cudaMemcpyHostToDevice));
    double endTransfer = getTime();
    printf("Host to Device transfer: %.6f seconds\n", endTransfer - startTransfer);

    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((img->width + blockSize.x - 1) / blockSize.x,
                  (img->height + blockSize.y - 1) / blockSize.y);
    
    int linearBlockSize = 256;
    int linearGridSize = (img->width * img->height + linearBlockSize - 1) / linearBlockSize;

    // Gaussian Blur
    double startBlur = getTime();
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_blurred, 
                                                 img->width, img->height, img->channels);
    CUDA_CHECK(cudaDeviceSynchronize());
    double endBlur = getTime();

    // RGB to Gray
    double startGray = getTime();
    rgbToGrayKernel<<<linearGridSize, linearBlockSize>>>(d_blurred, d_gray,
                                                          img->width, img->height, img->channels);
    CUDA_CHECK(cudaDeviceSynchronize());
    double endGray = getTime();

    // Sobel Edge Detection
    double startSobel = getTime();
    sobelKernel<<<gridSize, blockSize>>>(d_gray, d_edges, img->width, img->height);
    CUDA_CHECK(cudaDeviceSynchronize());
    double endSobel = getTime();

    // Copy result back to host
    unsigned char* edges = (unsigned char*)malloc(graySize);
    startTransfer = getTime();
    CUDA_CHECK(cudaMemcpy(edges, d_edges, graySize, cudaMemcpyDeviceToHost));
    endTransfer = getTime();
    printf("Device to Host transfer: %.6f seconds\n", endTransfer - startTransfer);

    // Save result
    Image result;
    result.width = img->width;
    result.height = img->height;
    result.channels = 1;
    result.data = edges;

    startIO = getTime();
    saveBMP(argv[2], &result);
    endIO = getTime();

    double totalCompute = (endBlur - startBlur) + (endGray - startGray) + (endSobel - startSobel);
    double endTotal = getTime();

    printf("Gaussian Blur: %.6f seconds\n", endBlur - startBlur);
    printf("RGB to Gray: %.6f seconds\n", endGray - startGray);
    printf("Sobel Edge: %.6f seconds\n", endSobel - startSobel);
    printf("Total Compute: %.6f seconds\n", totalCompute);
    printf("Save time: %.6f seconds\n", endIO - startIO);
    printf("Total time: %.6f seconds\n", endTotal - startTotal);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_edges));
    free(edges);
    freeImage(img);

    return 0;
}

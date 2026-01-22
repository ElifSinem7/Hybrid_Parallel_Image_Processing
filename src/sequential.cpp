#include "image_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    float kernel[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    float sum = 273.0f;

    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
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
    }
}

void rgbToGray(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    if (channels == 1) {
        memcpy(output, input, width * height);
        return;
    }

    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        output[i] = (unsigned char)(0.299f * input[idx + 2] + 
                                    0.587f * input[idx + 1] + 
                                    0.114f * input[idx]);
    }
}

void sobelEdgeDetection(unsigned char* input, unsigned char* output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int idx = (y + ky) * width + (x + kx);
                    sumX += input[idx] * Gx[ky + 1][kx + 1];
                    sumY += input[idx] * Gy[ky + 1][kx + 1];
                }
            }
            
            int magnitude = (int)sqrt((float)(sumX * sumX + sumY * sumY));
            output[y * width + x] = (magnitude > 255) ? 255 : magnitude;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input.bmp> <output.bmp>\n", argv[0]);
        return 1;
    }

    double startTotal = getTime();
    
    // Load image
    double startIO = getTime();
    Image* img = loadBMP(argv[1]);
    if (!img) return 1;
    double endIO = getTime();

    printf("Sequential Implementation\n");
    printf("Image: %dx%d, %d channels\n", img->width, img->height, img->channels);
    printf("Load time: %.6f seconds\n", endIO - startIO);

    int size = img->width * img->height * img->channels;
    int graySize = img->width * img->height;

    unsigned char* blurred = (unsigned char*)malloc(size);
    unsigned char* gray = (unsigned char*)malloc(graySize);
    unsigned char* edges = (unsigned char*)malloc(graySize);

    // Gaussian Blur
    double startBlur = getTime();
    gaussianBlur(img->data, blurred, img->width, img->height, img->channels);
    double endBlur = getTime();

    // RGB to Gray
    double startGray = getTime();
    rgbToGray(blurred, gray, img->width, img->height, img->channels);
    double endGray = getTime();

    // Sobel Edge Detection
    double startSobel = getTime();
    sobelEdgeDetection(gray, edges, img->width, img->height);
    double endSobel = getTime();

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

    free(blurred);
    free(gray);
    freeImage(img);

    return 0;
}

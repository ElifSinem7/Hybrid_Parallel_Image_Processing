#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <stdint.h>

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image;

#pragma pack(push, 1)
typedef struct {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPHeader;

typedef struct {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
} BMPInfoHeader;
#pragma pack(pop)

Image* loadBMP(const char* filename);
void saveBMP(const char* filename, Image* img);
void freeImage(Image* img);

#endif

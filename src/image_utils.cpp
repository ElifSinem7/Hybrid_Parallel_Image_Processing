#include "image_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Image* loadBMP(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    BMPHeader header;
    BMPInfoHeader infoHeader;

    fread(&header, sizeof(BMPHeader), 1, file);
    fread(&infoHeader, sizeof(BMPInfoHeader), 1, file);

    if (header.type != 0x4D42) {
        fprintf(stderr, "Error: Not a valid BMP file\n");
        fclose(file);
        return NULL;
    }

    Image* img = (Image*)malloc(sizeof(Image));
    img->width = infoHeader.width;
    img->height = infoHeader.height;
    img->channels = infoHeader.bitsPerPixel / 8;

    int rowSize = ((img->width * img->channels + 3) / 4) * 4;
    int dataSize = rowSize * img->height;
    
    unsigned char* tempData = (unsigned char*)malloc(dataSize);
    fseek(file, header.offset, SEEK_SET);
    fread(tempData, 1, dataSize, file);
    fclose(file);

    // BMP'ler ters çevrilmiş olarak saklanır, düzeltelim
    img->data = (unsigned char*)malloc(img->width * img->height * img->channels);
    for (int y = 0; y < img->height; y++) {
        memcpy(img->data + y * img->width * img->channels,
               tempData + (img->height - 1 - y) * rowSize,
               img->width * img->channels);
    }

    free(tempData);
    return img;
}

void saveBMP(const char* filename, Image* img) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return;
    }

    int rowSize = ((img->width * img->channels + 3) / 4) * 4;
    int dataSize = rowSize * img->height;

    BMPHeader header = {0};
    header.type = 0x4D42;
    header.size = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + dataSize;
    header.offset = sizeof(BMPHeader) + sizeof(BMPInfoHeader);

    BMPInfoHeader infoHeader = {0};
    infoHeader.size = sizeof(BMPInfoHeader);
    infoHeader.width = img->width;
    infoHeader.height = img->height;
    infoHeader.planes = 1;
    infoHeader.bitsPerPixel = img->channels * 8;
    infoHeader.compression = 0;
    infoHeader.imageSize = dataSize;

    fwrite(&header, sizeof(BMPHeader), 1, file);
    fwrite(&infoHeader, sizeof(BMPInfoHeader), 1, file);

    unsigned char* tempData = (unsigned char*)calloc(dataSize, 1);
    for (int y = 0; y < img->height; y++) {
        memcpy(tempData + (img->height - 1 - y) * rowSize,
               img->data + y * img->width * img->channels,
               img->width * img->channels);
    }

    fwrite(tempData, 1, dataSize, file);
    free(tempData);
    fclose(file);
}

void freeImage(Image* img) {
    if (img) {
        if (img->data) free(img->data);
        free(img);
    }
}

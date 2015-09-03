// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>

int* load_correct(char* filename, int width, int height){
    int* correct = (int*)malloc(sizeof(int)*width*height);
    
    FILE* file;
    file = fopen(filename, "rb");
    if(!file){
        fprintf(stderr,"did not find file\n");
    }
    for(int i = 0; i < width*height; i++){
        fread(&correct[i], sizeof(int), 1, file);
    }
    fclose(file);
    
    return correct;
}

void write_image_raw(char* filename, int* image, int width, int height){
    FILE* file;
    file = fopen(filename, "wb");
    for(int i = 0; i < width*height; i++){
        fwrite(&image[i], sizeof(int), 1, file);
    }
    fclose(file);
}

float* load_correct_float(char* filename, int width, int height){
    float* correct = (float*)malloc(sizeof(float)*width*height);

    FILE* file;
    file = fopen(filename, "rb");
    if(!file){
        fprintf(stderr,"did not find file\n");
    }
    for(int i = 0; i < width*height; i++){
        fread(&correct[i], sizeof(float), 1, file);
    }
    fclose(file);

    return correct;
}

void write_image_raw_float(char* filename, float* image, int width, int height){
    FILE* file;
    file = fopen(filename, "wb");
    for(int i = 0; i < width*height; i++){
        fwrite(&image[i], sizeof(float), 1, file);
    }
    fclose(file);
}

void write_raw_buffer(char* filename, unsigned char* buffer, int length){
    FILE* file;
    file = fopen(filename, "wb");
    for(int i = 0; i < length; i++){
        fwrite(&buffer[i], sizeof(unsigned char), 1, file);
    }
    fclose(file);
}

unsigned char* load_raw_buffer(char* filename, int size){
    unsigned char* correct = (unsigned char*)malloc(sizeof(unsigned char)*size);
    
    FILE* file;
    file = fopen(filename, "rb");
    if(!file){
        fprintf(stderr,"did not find file\n");
    }
    for(int i = 0; i < size; i++){
        fread(&correct[i], sizeof(unsigned char), 1, file);
    }
    fclose(file);
    
    return correct;
}

void write_ppm(int* image, int width, int height){
    printf("P3\n");
    printf("%d %d\n", width, height);
    printf("%d\n", 255);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            unsigned char* rgba = (unsigned char*)&image[i*width + j];
            printf("%u ", rgba[0]);
            printf("%d ", rgba[1]);
            printf("%d ", rgba[2]);
            printf("\n");
        }
    }
}

void write_ppm_bw(int* image, int width, int height){
    printf("P3\n");
    printf("%d %d\n", width, height);
    printf("%d\n", 255);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            //unsigned char* rgba = (unsigned char*)&image[i*width + j];
            printf("%u ", image[i*width+j]);
            printf("%d ", image[i*width+j]);
            printf("%d ", image[i*width+j]);
            printf("\n");
        }
    }
}

void write_ppm_uchar(unsigned char* image, int width, int height){
    printf("P3\n");
    printf("%d %d\n", width, height);
    printf("%d\n", 255);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            printf("%u ", image[i*width+j]);
            printf("%d ", image[i*width+j]);
            printf("%d ", image[i*width+j]);
            printf("\n");
        }
    }
}

void write_ppm_crossection_uchar(unsigned char* volume, int width, int height, int depth){
    printf("P3\n");
    printf("%d %d\n", width, height);
    printf("%d\n", 255);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
        
            int index = i*width*height + i*width + j;
            printf("%u ", volume[index]);
            printf("%d ", volume[index]);
            printf("%d ", volume[index]);
            printf("\n");
        }
    }
}
    

int* load_ppm(const char* filename, int* width, int* height){
    FILE* file = fopen(filename, "r");
    char magic[2];
    int max_color;
    
    fscanf(file, "%s %d %d %d ", magic, width, height, &max_color);
    
    //printf("%s, %d, %d, %d\n",magic, width, height, max_color);
    
    int* image = (int*)malloc(sizeof(int)*(*width)*(*height));
    unsigned char* scanline = (unsigned char*)malloc(sizeof(unsigned char) * *width * 3);
    
    for(int i = 0; i < *height; i++){
        fread((void*)scanline, *width*3, 1, file);
        
        for(int j = 0; j < *width; j++){
            image[i**width + j] =  ((unsigned int)(0) | ((unsigned int)(scanline[j*3+2])<<16) | ((unsigned int)(scanline[j*3+1])<<8) | (unsigned int)(scanline[j*3]));
        }
    }
    
    return image;
}

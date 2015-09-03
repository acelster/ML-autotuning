// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#ifndef IO
#define IO

void write_image_raw_float(char* filename, float* image, int width, int height);
float* load_correct_float(char* filename, int width, int height);
void write_image_raw(char* filename, int* image, int width, int height);
int* load_correct(char* filename, int width, int height);
void write_ppm(int* image, int width, int height);
void write_ppm_bw(int* image, int width, int height);
void write_ppm_uchar(unsigned char* image, int width, int height);
void write_ppm_crossection_uchar(unsigned char* volume, int width, int height, int depth);
int* load_ppm(const char* filename, int* width, int* height);
unsigned char* load_raw_buffer(char* filename, int size);
void write_raw_buffer(char* filename, unsigned char* buffer, int length);

#endif

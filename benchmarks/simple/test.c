// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

const int MODE_ALL = 0;
const int MODE_TIMED = 1;

int* parse_file(char* input_file, char* output_file, int n_samples, int n_parameters, int mode, float* timeThreshold){

    FILE* file = fopen(input_file, "r");
    if(file == NULL){
        fprintf(stderr, "Unable to open file, exiting\n");
        exit(-1);
    }

    int a,b,c,d;
    if(mode == MODE_ALL){
    fscanf(file, "%d %d\n", &a, &b);
    }
    else{
        fscanf(file,"%d %d %f %d %d\n",&a, &b, timeThreshold, &c, &d);
    }

    int* configs = (int*)malloc(sizeof(int)*n_samples*n_parameters);
    
    for(int i = 0; i < n_samples; i++){
        fscanf(file, "%d %d %d %d\n", 
               &configs[i*n_parameters], 
               &configs[i*n_parameters+1],
               &configs[i*n_parameters+2],
               &configs[i*n_parameters+3]);
    }

    fclose(file);
    
    return configs;
}


int main(int argc, char** argv){
    
    if(argc != 5){
        printf("Useage: %s infile outfile n_samples mode\n", argv[0]);
        exit(0);
    }
    
    srand(time(NULL));
    
    char* input_file = argv[1];
    char* output_file = argv[2];
    int n_samples = atoi(argv[3]);
    int mode = atoi(argv[4]);
    int n_parameters = 4;
    
    float timeThreshold;
    int* configs = parse_file(input_file, output_file, n_samples, n_parameters, mode, &timeThreshold);
    
    
    FILE* file = fopen(output_file, "w+");
    
    for(int i = 0; i < n_samples; i++){
        int a = configs[i*n_parameters];
        int b = configs[i*n_parameters+1];
        int c = configs[i*n_parameters+2];
        int d = configs[i*n_parameters+3];
        
        float f = abs(a-1) + abs(b-2) + abs(c-3) + abs(d-4);

        if(mode == MODE_TIMED && f > timeThreshold && i > 0){
            break;
        }
        
        fprintf(file, "%d %d %d %d %f\n", a, b, c, d, f);
    }
    
    fclose(file);
}

// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>


int* parse_file(char* input_file, char* output_file, int n_samples, int n_parameters){

    FILE* file = fopen(input_file, "r");
    if(file == NULL){
        fprintf(stderr, "Unable to open file, exiting\n");
        exit(-1);
    }

    int a,b;
    fscanf(file, "%d %d\n", &a, &b);

    int* configs = (int*)malloc(sizeof(int)*n_samples*n_parameters);
    
    for(int i = 0; i < n_samples; i++){
        fscanf(file, "%d %d %d\n", 
               &configs[i*n_parameters], 
               &configs[i*n_parameters+1],
               &configs[i*n_parameters+2]);
    }

    fclose(file);
    
    return configs;
}

double get_time(struct timeval start, struct timeval end){
    long int ms = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    return ms/1e6;
}


int main(int argc, char** argv){
    
    if(argc != 4){
        printf("Useage: %s infile outfile n_samples\n", argv[0]);
        exit(0);
    }
    
    srand(time(NULL));
    
    char* input_file = argv[1];
    char* output_file = argv[2];
    int n_samples = atoi(argv[3]);
    int n_parameters = 3;
    
    int* configs = parse_file(input_file, output_file, n_samples, n_parameters);
    
    int k = 300;
    int l = 200;
    int m = 350;
    double* a = (double*)malloc(sizeof(double)*k*l);
    double* b = (double*)malloc(sizeof(double)*m*l);
    double* c = (double*)calloc(m*k,sizeof(double));
    //double* d = (double*)calloc(m*k,sizeof(double));
    
    for(int i = 0; i < k*l; i++){
        a[i] = (float)rand()/RAND_MAX;
    }
    for(int i = 0; i < m*l; i++){
        b[i] = (float)rand()/RAND_MAX;
    }
    
    FILE* file = fopen(output_file, "w+");
    
    for(int i = 0; i < n_samples; i++){
        int bk = configs[i*n_parameters];
        int bl = configs[i*n_parameters+1];
        int bm = configs[i*n_parameters+2];
        
        bk++;
        bl++;
        bm++;
        
        struct timeval start, end;
        
        gettimeofday(&start, NULL);
        
        for(int xx = 0; xx < m; xx += bm){
            for(int yy = 0; yy < k; yy += bk){
                for(int zz = 0; zz < l; zz += bl){
                    
                    for(int x = xx; x < m && x < xx + bm; x++){
                        for(int y = yy; y < k && y < yy + bk; y++){
                            for(int z = zz; z < l && z < zz+bl; z++){
                                c[y*m + x] += a[y*l +z]*b[z*m + x];
                            }
                        }
                    }
                }
            }
        }
        gettimeofday(&end, NULL);
        double time = get_time(start, end);
        
        /*
        for(int x = 0; x < m; x++){
            for(int y = 0; y < k; y++){
                for(int z = 0; z < l; z++){
                    d[y*m + x] += a[y*l + z]*b[z*m + x];
                }
            }
        }
        
        for(int i= 0; i < m*k; i++){
            if( abs(c[i] - d[i]) > 1e-6){
                printf("%d, %f, %f\n", i, c[i], d[i]);
            }
        }
        */
        
        fprintf(file, "%d %d %d %f\n", bk, bl, bm, time);
    }
    
    fclose(file);
}
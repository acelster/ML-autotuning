// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <getopt.h>

#include "clutil.h"

static int permutation_start = 0;
static char* filename = NULL;
static int n_iterations = 0;
static int use_seeding = 0;
static int ignore_crashes = 0;
static int self_test = 0;
static char* correct_file = NULL;
static char* output_file = NULL;
static char* device = NULL;
static int use_time_threshold = 0;

//These could be moved
static float time_threshold = 0.0;
static int min_second_stage = 0;
static int max_second_stage = 0;

static const char* help = 
"Options: \n \
-h              Display this message and exit \n \
-i <arg>        Jump start to this configuration \n \
-f <file>       Input file with configurations \n \
-n <arg>        Number of configurations \n \
-r              Use second stage time threshold from file \n \
-s              Seed random number generator \n \
-m              Ignore crashes when counting \n \
-t              Self test \n \
-c <file>       Correct file \n \
-w <file>       Output file \n \
-l              List all available OpenCL devices and exit \n \
-d <arg>        Select OpenCL device \n \
\n";

void print_help(int argc, char** argv){
    printf("Useage: %s [options]\n\n%s", argv[0], help);
    exit(0);
}

void parse_args(int argc, char** argv){
    
    int c;
    while( (c = getopt(argc, argv, "htc:i:f:n:w:smld:r")) != -1){
        switch (c) {
            case 'h':
                print_help(argc, argv);
                break;
            case 'i':
                permutation_start = atoi(optarg);
                break;
            case 'f':
                filename = optarg;
                break;
            case 'n':
                n_iterations = atoi(optarg);
                break;
            case 's':
                use_seeding = 1;
                break;
            case 'm':
                ignore_crashes = 1;
                break;
            case 't':
                self_test = 1;
                break;
            case 'c':
                correct_file = optarg;
                break;
            case 'w':
                output_file = optarg;
                break;
            case 'l':
                list_all_devices();
                exit(1);
                break;
            case 'd':
                device = optarg;
                break;
            case 'r':
                use_time_threshold = 1;
                break;
            default:
                break;
        }
    }
    
    //TODO remove this
    if(filename == NULL && n_iterations == 0 && self_test == 0){
        printf("No iterations or inputfile specified.\nExiting\n");
        exit(-1);
    }
}

//TODO reevaluate this desing
float get_time_threshold(){
    return time_threshold;
}

int get_min_second_stage(){
    return min_second_stage;
}

int get_max_second_stage(){
    return max_second_stage;
}

int get_use_time_threshold(){
    return use_time_threshold;
}

char* get_output_file(){
    return output_file;
}

char* get_correct_file(){
    return correct_file;
}

int perform_self_test(){
    return self_test;
}

int get_use_seeding(){
	return use_seeding;
}

int ignore_crashes_when_counting(){
	return ignore_crashes;
}

int get_n_run_configurations_arg(){
	return n_iterations;
}

int get_start_iteration(){
	return permutation_start;
}

int read_from_file(){
	return filename != NULL;
}

int parse_num(int argc, char** argv){
    return atoi(argv[2]);
}

cl_device_id get_selected_device(){
    if(device == NULL){
        return get_device_by_id(0,0);
    }
    if(device[0] == 'g' || device[0] == 'G'){
        return get_device(CL_DEVICE_TYPE_GPU);
    }
    if(device[0] == 'c' || device[0] == 'C'){
        return get_device(CL_DEVICE_TYPE_CPU);
    }
    
    return get_device_by_id(device[0]-48, device[2]-48);
}

int* parse_file(int argc, char** argv, int* n, int* e, int* limits, int n_parameters){

	int* cumulutative = (int*)malloc(sizeof(int)* n_parameters);
	cumulutative[n_parameters-1] = 1;
	for(int i = n_parameters-2; i >= 0; i--){
	    cumulutative[i] = cumulutative[i+1]*limits[i+1];
	}

    FILE* file = fopen(argv[2], "r");
    if(file == NULL){
    	fprintf(stderr, "Unable to open file, exiting\n");
    	exit(-1);
    }

    int n_lines = 0, n_entries = 0;

    if(use_time_threshold){
        fscanf(file, "%d %d %f %d %d\n", &n_lines, &n_entries, &time_threshold, &min_second_stage, &max_second_stage);
    }
    else{
        fscanf(file, "%d %d\n", &n_lines, &n_entries);
    }

    int * configs = (int*)malloc(sizeof(int)*n_lines);
    int temp;
    int value;
    for(int l = 0; l < n_lines; l++){
    	value = 0;
        for(int f = 0; f < n_entries; f++){
            fscanf(file," %d ", &temp);
            value += temp * cumulutative[f];
        }
        configs[l] = value;
    }
    fclose(file);
    
    *n = n_lines;
    *e = n_entries;
    
    return configs;
}

/*
int main(int argc, char** argv){
    parse_args(argc, argv);

    printf("%d %d %s\n", permutation_start, n_iterations, filename);
}
*/


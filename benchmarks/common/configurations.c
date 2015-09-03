// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "parser.h"

int* create_configurations(int* limits, int n_parameters, int argc, char** argv, int* n_run_configurations, int* n_total_configurations){

	if(read_from_file()){
		int n, e;
		int* configurations = parse_file(argc, argv,&n, &e, limits, n_parameters);
		*n_run_configurations = n;
		*n_total_configurations = n;
		return configurations;
	}

    int n_combinations = 1;
    for(int i = 0; i < n_parameters; i++){
        n_combinations *= limits[i];
    }

    int* configurations = (int*)malloc(sizeof(int)*n_combinations);
    for(int i = 0; i < n_combinations; i++){
        configurations[i] = i;
    }
    
    if(get_use_seeding()){
    	srand(time(NULL));
    }

    for(int i = n_combinations-1; i >= 1; i--){
        int index = rand()%i;
        int temp = configurations[i];
        configurations[i] = configurations[index];
        configurations[index] = temp;
    }
    *n_total_configurations = n_combinations;

    int n_run_configurations_arg = get_n_run_configurations_arg();
    if(n_run_configurations_arg == 0){
    	*n_run_configurations = n_combinations;
    }
    else{
    	*n_run_configurations = n_run_configurations_arg;
    }
    return configurations;

}

int* get_config_for_number(int n, int* limits, int n_parameters){
    
    int* config = (int*)calloc(sizeof(int), n_parameters);
    int* cumulutative = (int*)malloc(sizeof(int)* n_parameters);
    
    cumulutative[n_parameters-1] = 1;
    for(int i = n_parameters-2; i >= 0; i--){
        cumulutative[i] = cumulutative[i+1]*limits[i+1];
    }
    
    for(int i = 0; i < n_parameters; i++){
        config[i] = n/cumulutative[i];
        n = n - (config[i] * cumulutative[i]);
    }
    free(cumulutative);
    return config;
}

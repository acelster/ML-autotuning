// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#ifndef PARSER
#define PARSER

#include <CL/cl.h>

int parse_args(int argc, char** argv);
int read_from_file();
int perform_self_test();
int get_n_run_configurations_arg();
int get_start_iteration();
int get_use_seeding();
int ignore_crashes_when_counting();
int parse_num(int argc, char** argv);
char* get_correct_file();
char* get_output_file();
int* parse_file(int argc, char** argv, int* n, int* e, int* limits, int n_parameters);
cl_device_id get_selected_device();
float get_time_threshold();
int get_min_second_stage();
int get_max_second_stage();
int get_use_time_threshold();
        
#endif

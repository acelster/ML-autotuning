// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#ifndef CONFIGURATIONS_HEADER 
#define CONFIGURATIONS_HEADER 
int* get_config_for_number(int n, int* limits, int n_parameters);
int* create_configurations(int* limits, int n_parameters, int argc, char** argv, int* n_run_permutations, int* n_total_permutations);

#endif

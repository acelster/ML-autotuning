// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#ifndef CLUTIL_H
#define CLUTIL_H
#include <CL/cl.h>

void list_all_devices();
const char *clErrorStr(cl_int err);
void clError(char *s, cl_int err);
void printPlatformInfo(cl_platform_id platform);
void printDeviceInfo(cl_device_id device);
int invalid_work_group_size(cl_device_id id, cl_kernel kernel, int dim, const size_t* local_work_size, const size_t* global_work_size);
int invalid_work_group_size_static(cl_device_id id, int dim, const size_t* local_work_size, const size_t* global_work_size);
cl_device_id get_device(cl_device_type device_type);
cl_device_id get_device_by_id(int platform_index, int device_index);
cl_kernel buildKernel(char* sourceFile, char* kernelName, char* options, cl_context context, cl_device_id device, cl_int* error);

#endif

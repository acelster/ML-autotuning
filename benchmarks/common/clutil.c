// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include "clutil.h"
#include <CL/cl.h>
#include <stdio.h>

const char *clErrorStr(cl_int err) {
	switch (err) {
	case CL_SUCCESS:                          return "Success!";
	case CL_DEVICE_NOT_FOUND:                 return "Device not found.";
	case CL_DEVICE_NOT_AVAILABLE:             return "Device not available";
	case CL_COMPILER_NOT_AVAILABLE:           return "Compiler not available";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "Memory object allocation failure";
	case CL_OUT_OF_RESOURCES:                 return "Out of resources";
	case CL_OUT_OF_HOST_MEMORY:               return "Out of host memory";
	case CL_PROFILING_INFO_NOT_AVAILABLE:     return "Profiling information not available";
	case CL_MEM_COPY_OVERLAP:                 return "Memory copy overlap";
	case CL_IMAGE_FORMAT_MISMATCH:            return "Image format mismatch";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return "Image format not supported";
	case CL_BUILD_PROGRAM_FAILURE:            return "Program build failure";
	case CL_MAP_FAILURE:                      return "Map failure";
	case CL_INVALID_VALUE:                    return "Invalid value";
	case CL_INVALID_DEVICE_TYPE:              return "Invalid device type";
	case CL_INVALID_PLATFORM:                 return "Invalid platform";
	case CL_INVALID_DEVICE:                   return "Invalid device";
	case CL_INVALID_CONTEXT:                  return "Invalid context";
	case CL_INVALID_QUEUE_PROPERTIES:         return "Invalid queue properties";
	case CL_INVALID_COMMAND_QUEUE:            return "Invalid command queue";
	case CL_INVALID_HOST_PTR:                 return "Invalid host pointer";
	case CL_INVALID_MEM_OBJECT:               return "Invalid memory object";
	case CL_INVALID_IMAGE_SIZE:               return "Invalid image size";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return "Invalid image format descriptor";
	case CL_INVALID_SAMPLER:                  return "Invalid sampler";
	case CL_INVALID_BINARY:                   return "Invalid binary";
	case CL_INVALID_BUILD_OPTIONS:            return "Invalid build options";
	case CL_INVALID_PROGRAM:                  return "Invalid program";
	case CL_INVALID_PROGRAM_EXECUTABLE:       return "Invalid program executable";
	case CL_INVALID_KERNEL_NAME:              return "Invalid kernel name";
	case CL_INVALID_KERNEL_DEFINITION:        return "Invalid kernel definition";
	case CL_INVALID_KERNEL:                   return "Invalid kernel";
	case CL_INVALID_ARG_INDEX:                return "Invalid argument index";
	case CL_INVALID_ARG_VALUE:                return "Invalid argument value";
	case CL_INVALID_ARG_SIZE:                 return "Invalid argument size";
	case CL_INVALID_KERNEL_ARGS:              return "Invalid kernel arguments";
	case CL_INVALID_WORK_DIMENSION:           return "Invalid work dimension";
	case CL_INVALID_WORK_GROUP_SIZE:          return "Invalid work group size";
	case CL_INVALID_WORK_ITEM_SIZE:           return "Invalid work item size";
	case CL_INVALID_GLOBAL_OFFSET:            return "Invalid global offset";
	case CL_INVALID_EVENT_WAIT_LIST:          return "Invalid event wait list";
	case CL_INVALID_EVENT:                    return "Invalid event";
	case CL_INVALID_OPERATION:                return "Invalid operation";
	case CL_INVALID_GL_OBJECT:                return "Invalid OpenGL object";
	case CL_INVALID_BUFFER_SIZE:              return "Invalid buffer size";
	case CL_INVALID_MIP_LEVEL:                return "Invalid mip-map level";
	default:                                  return "Unknown";
	}
}
void clError(char *s, cl_int err) {
    if(err != CL_SUCCESS){
	    fprintf(stderr,"%s: %s\n",s,clErrorStr(err));
	    //exit(1);
    }
}

void printPlatformInfo(cl_platform_id platform){
    cl_int err;
    char paramValue[100];
    size_t returnSize;

    printf("===== Platform info ====\n");

    err = clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, 100, paramValue, &returnSize);
    printf("Platform profile:\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 100, paramValue, &returnSize);
    printf("Platform version:\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 100, paramValue, &returnSize);
    printf("Platform name:\t\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 100, paramValue, &returnSize);
    printf("Platform vendor:\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 100, paramValue, &returnSize);
    printf("Platform extensions:\t");
    if(returnSize > 0){
        printf("%s\n", paramValue);
    }
    else{
        printf("(none)");
    }

    clError("Couldn't get device info", err);

    printf("\n\n");
}

void printDeviceInfo(cl_device_id device){
    cl_int err;
    char paramValueString[100];
    cl_ulong paramValueUlong;
    cl_uint paramValueUint;
    size_t paramValueSizet;
    size_t returnSize;

    printf("==== Device info ====\n");

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 100, paramValueString, &returnSize);
    printf("Device name:\t\t%s\n", paramValueString);

    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &paramValueUlong, &returnSize);
    printf("Device global memory:\t%lu\n", paramValueUlong);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &paramValueUint, &returnSize);
    printf("Device max frequency:\t%u\n", paramValueUint);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &paramValueUint, &returnSize);
    printf("Device compute units:\t%u\n", paramValueUint);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &paramValueUlong, &returnSize);
    printf("Max size of memory allocation:\t%lu\n", paramValueUlong);
    
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &paramValueUlong, &returnSize);
    printf("Local memory size:\t%lu\n", paramValueUlong);

    err = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &paramValueUint, &returnSize);
    printf("Device memory alignment:\t%u\n", paramValueUint);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &paramValueSizet, &returnSize);
    printf("Max work group size:\t%zu\n", paramValueSizet);
    
    size_t dims[3];
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, &returnSize);
    printf("Max work size dimensions:  %zu, %zu, %zu\n", dims[0], dims[1], dims[2]);
    
    clError("Couldn't get device info", err);


    printf("\n\n");
}

void list_all_devices(){
    cl_int error;
    cl_uint n_platforms;
    
    error = clGetPlatformIDs(0, NULL, &n_platforms);
    
    if(n_platforms == 0){
        printf("No OpenCL platforms found\n");
    }
    
    cl_platform_id* all_platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*n_platforms);
    error = clGetPlatformIDs(n_platforms, all_platforms, NULL);
    
    char string[100];
    char string2[100];
    size_t returnSize;
    cl_uint n_devices;
    for(int i = 0; i < n_platforms; i++){
        
        //printPlatformInfo(all_platforms[i]); 
        clGetPlatformInfo(all_platforms[i], CL_PLATFORM_NAME, 100, string, &returnSize);
        clGetPlatformInfo(all_platforms[i], CL_PLATFORM_VERSION, 100, string2, &returnSize);
        printf("Platform %d: %s (%s)\n", i, string, string2);
        
        error = clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
        cl_device_id* all_devices = (cl_device_id*)malloc(sizeof(cl_device_id)*n_devices);
        error = clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL, n_devices, all_devices, NULL);
        
        for(int j = 0; j < n_devices; j++){
            //printDeviceInfo(all_devices[j]);
            clGetDeviceInfo(all_devices[j], CL_DEVICE_NAME, 100, string, &returnSize);
            printf("\tDevice %d: %s\n", j, string);
        }
        printf("\n");
    }
    clError("list all devices", error);
}

cl_device_id get_device_by_id(int platform_index, int device_index){
    cl_int error;
    cl_uint n_platforms;
    
    error = clGetPlatformIDs(0, NULL, &n_platforms);
    
    if(n_platforms == 0){
        printf("No OpenCL platforms found\n");
    }
    
    cl_platform_id* all_platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*n_platforms);
    error = clGetPlatformIDs(n_platforms, all_platforms, NULL);
    
    cl_uint n_devices;
        
        
    error = clGetDeviceIDs(all_platforms[platform_index], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
    cl_device_id* all_devices = (cl_device_id*)malloc(sizeof(cl_device_id)*n_devices);
    error = clGetDeviceIDs(all_platforms[platform_index], CL_DEVICE_TYPE_ALL, n_devices, all_devices, NULL);
    
    clError("get device by id", error);
    return all_devices[device_index];
}

cl_device_id get_device(cl_device_type device_type){
    
    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_uint n_platforms;
    
    error = clGetPlatformIDs(0, NULL, &n_platforms);
    
    if(n_platforms == 0){
        fprintf(stderr, "No OpenCL platforms found\n");
        return NULL;
    }
    
    cl_platform_id* all_platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*n_platforms);
    error = clGetPlatformIDs(n_platforms, all_platforms, NULL);
    
    cl_uint n_devices;
    int platform_to_use = -1;
    for(int i = 0; i < n_platforms; i++){
        error = clGetDeviceIDs(all_platforms[i], device_type, 0, NULL, &n_devices);
        if(n_devices > 0){
            platform_to_use = i;
            break;
        }
    }
    
    if(platform_to_use == -1){
        fprintf(stderr, "No platform with the requested device\n");
        return NULL;
    }

    platform = all_platforms[platform_to_use];
    
    cl_device_id* all_devices = (cl_device_id*)malloc(sizeof(cl_device_id)*n_devices);
    error = clGetDeviceIDs(platform, device_type, n_devices, all_devices, NULL);
    device = all_devices[0];
    
    free(all_devices);
    free(all_platforms);
    
    //printPlatformInfo(platform);
    //printDeviceInfo(device);
    
    clError("Couldn't get device", error);

    return device;
}

int invalid_work_group_size(cl_device_id device, cl_kernel kernel, int dim, const size_t* local_work_size, const size_t* global_work_size){
    
    
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    
    size_t max_work_item_sizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
    
    size_t kernel_max_work_group_size;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_max_work_group_size), &kernel_max_work_group_size, NULL);
    
    
    //printf("%zu, %zu\n", max_work_group_size, kernel_max_work_group_size);
    int invalid = 0;
    size_t total_size = 1;
    for(int i = 0; i < dim; i++){
        if(local_work_size[i] > global_work_size[i])
            invalid = 1;
        if(local_work_size[i] > max_work_item_sizes[i])
            invalid = 1;
        total_size *= local_work_size[i];
    }
    if(total_size > max_work_group_size)
        invalid = 1;
    if(total_size > kernel_max_work_group_size)
        invalid = 1;
    
    return invalid;
}

int invalid_work_group_size_static(cl_device_id device, int dim, const size_t* local_work_size, const size_t* global_work_size){
    
    
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    
    size_t max_work_item_sizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
    
    
    int invalid = 0;
    size_t total_size = 1;
    for(int i = 0; i < dim; i++){
        if(local_work_size[i] > global_work_size[i])
            invalid = 1;
        if(local_work_size[i] > max_work_item_sizes[i])
            invalid = 1;
        total_size *= local_work_size[i];
    }
    if(total_size > max_work_group_size)
        invalid = 1;
    
    return invalid;
}

char *load_program_source(const char *s) {
	char *t;
	size_t len;
	FILE *f = fopen(s, "r");
	if(NULL== f){
        fprintf(stderr,"couldn't open file");
        exit(0);
    }
	fseek(f,0,SEEK_END);
	len=ftell(f);
	fseek(f,0,SEEK_SET);
	t=malloc(len+1);
	fread(t,len,1,f);
	t[len]=0;
	fclose(f);
	return t;
}

cl_kernel buildKernel(char* sourceFile, char* kernelName, char* options, cl_context context, cl_device_id device, cl_int* error){
    cl_int err;
    
    char* source = load_program_source(sourceFile);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
    clError("Error creating program",err);
    
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if(CL_SUCCESS != err) {
        static char s[1048576];
        size_t len;
        //clError("Error building program", err);
        fprintf(stderr,"Error building program\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(s), s, &len);
        fprintf(stderr,"Build log:\n%s\n", s);
        free(source);
        clReleaseProgram(program);
        *error = err;
        return NULL;
    }
    
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    clError("Error creating kernel",err);
    free(source);
    clReleaseProgram(program);

    return kernel;
}

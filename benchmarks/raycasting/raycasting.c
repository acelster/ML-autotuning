// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <CL/cl.h>

#include "../common/clutil.h"

#include "../common/configurations.h"
#include "../common/io.h"
#include "../common/parser.h"

//Problem parameters
#define IMAGE_HEIGHT (512)
#define IMAGE_WIDTH (512)
#define DATA_DIM 128
#define TRANSFER_FUNC_SIZE 128
#define USE_TRILINEAR 1

//Tuning parameters

int LOCAL_SIZE_X =              0;
int LOCAL_SIZE_Y =              1;
int ELEMENTS_PER_THREAD_X =     2;
int ELEMENTS_PER_THREAD_Y =     3;
int USE_TEXTURE_DATA =          4;
int USE_TEXTURE_TRANSFER =      5;
int USE_SHARED_TRANSFER =       6;
int USE_CONSTANT_TRANSFER =     7;
int INTERLEAVED =               8;
int UNROLL_FACTOR =             9;

int global_config[] = {3,3,1,1,1,1,0,0,0,0};
int param_limits[] = {8,8,8,8,2,2,2,2,2,5}; //Or, rather, the limit + 1
int n_parameters = 10;


typedef struct{
    float x;
    float y;
    float z;
    float w;
} float4;


float func(int x, int y, int z){
    
    //x = (x/32)*32;
    //y = (y/32)*32;
    //z = (z/32)*32;
    
    float value =  (rand() % 10)/100.0;
    
    int x1 = 150/2;
    int y1 = 200/2;
    int z1 = 50/2;
    float dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));
    
    if(dist < 50/2){
        value  = 0.6;
    }
    
    x1 = 50/2;
    y1 = 100/2;
    z1 = 200/2;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));
    
    if(dist < 50/2){
        value = 0.5;
    }
    
    x1 = 500/2;
    y1 = 75/2;
    z1 = 50/2;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));
    
    if(dist < 30/2){
        value = 0.9;
    }
    
    x1 = 200/2;
    y1 = 75/2;
    z1 = 20/2;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));
    
    if(dist < 30/2){
        value = 0.7;
    }
    
    x1 = 75/2;
    y1 = 200/2;
    z1 = 175/2;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));
    
    if(dist < 50/2){
        value = 0.8;
    }
    
    
    return value;
}


float* create_data(){
    float* data = (float*)malloc(sizeof(float)*DATA_DIM*DATA_DIM*DATA_DIM);
    
    for(int x = 0; x < DATA_DIM; x++){
        for(int y = 0; y < DATA_DIM; y++){
            for(int z = 0; z < DATA_DIM; z++){
                data[z*DATA_DIM*DATA_DIM + y*DATA_DIM + x] = func(x,y,z);
            }
        }
    }
    return data;
}


void print_image(int* image){
    printf("P3\n");
    printf("%d %d\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("%d\n", 255);
    for(int i = 0; i < IMAGE_HEIGHT; i++){
        for(int j = 0; j < IMAGE_WIDTH; j++){
            unsigned char* rgba = (unsigned char*)&image[i*IMAGE_WIDTH + j];
            printf("%u ", rgba[0]);
            printf("%d ", rgba[1]);
            printf("%d ", rgba[2]);
            printf("\n");
        }
    }
}


int check_image(int* correct, int* image){
    
    if(correct == NULL || image == NULL){
        return 1;
    }
    
    int n_errors = 0;
    
    for(int i = 0; i < IMAGE_WIDTH*IMAGE_HEIGHT; i++){
        int ri = (image[i] & 0xFF000000) >> 24;
        int gi = (image[i] & 0x00FF0000) >> 16;
        int bi = (image[i] & 0x0000FF00) >> 8;
        int ai = (image[i] & 0x000000FF);
        
        int rc = (correct[i] & 0xFF000000) >> 24;
        int gc = (correct[i] & 0x00FF0000) >> 16;
        int bc = (correct[i] & 0x0000FF00) >> 8;
        int ac = (correct[i] & 0x000000FF);
        
        int dr = abs(ri - rc);
        int dg = abs(gi - gc);
        int db = abs(bi - bc);
        int da = abs(ai - ac);
        
        if(dr > 1 || dg > 1 || db > 1 || da > 1){
            if(n_errors < 10){
                fprintf(stderr,"Error at : %d, expected %x, found %x\n", i, correct[i], image[i]);
                
            }
            n_errors++;
        }
    }
    if(n_errors > 0){
        fprintf(stderr,"%d errors in total\n", n_errors-10);
    }
    
    return n_errors == 0;
}


cl_float4* create_transfer(){
    cl_float4* transfer = (cl_float4*)calloc(sizeof(cl_float4),TRANSFER_FUNC_SIZE);
    for(int i = 0; i < TRANSFER_FUNC_SIZE; i++){
        
        if(i < 20){
            transfer[i].s[3] = 0.001;
        }
        else{
            transfer[i].s[3] = 0.0035;
        
        }
        
        transfer[i].s[0] = (sin(1+(i/(float)TRANSFER_FUNC_SIZE)*4*3.14)+1)/2.0;
        transfer[i].s[1] = (sin(3.14 + (i/(float)TRANSFER_FUNC_SIZE)*4*3.14)+1)/2.0;
        transfer[i].s[2] = (sin( 0.1 + (i/(float)TRANSFER_FUNC_SIZE)*4*3.14)+1)/2.0;
    }
    
    return transfer;
}


char* timestamp(){
    time_t ltime;
    ltime=time(NULL);
    char* ts = malloc(50);
    sprintf(ts, "%s",asctime( localtime(&ltime) ) );
    return ts;
}


void print_comment(cl_device_id device, char** argv){
    printf("# %s\n", argv[0]);
    
    time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
    printf("# %s",asctime( localtime(&ltime) ) );
    
    char name[100];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 100, name, NULL);
    printf("# %s\n", name);
    printf("\n");
    
    printf("# IMAGE_HEIGHT %d\n", IMAGE_HEIGHT);
    printf("# IMAGE_WIDTH %d\n", IMAGE_WIDTH);
    printf("# DATA_DIM %d\n", DATA_DIM);
    printf("# TRANSFER_FUNC_SIZE %d\n", TRANSFER_FUNC_SIZE);
    printf("# USE_TRILINEAR %d\n", USE_TRILINEAR);
    printf("\n");
}


double raycast_ocl(float* data_host, cl_float4* transfer_host, int* image_host, cl_device_id device, int* config){
    
    int lwsx = pow(2, config[LOCAL_SIZE_X]);
    int lwsy = pow(2, config[LOCAL_SIZE_Y]);
    int eptx = pow(2, config[ELEMENTS_PER_THREAD_X]);
    int epty = pow(2, config[ELEMENTS_PER_THREAD_Y]);
    const size_t local_work_size[2] = {lwsx,lwsy};
    const size_t global_work_size[2] = {(IMAGE_WIDTH/eptx),IMAGE_HEIGHT/epty};
    
    if(invalid_work_group_size_static(device, 2, local_work_size, global_work_size)){
        return -1;
    }
    
    char options_buffer [300];
    cl_int error;
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    clError("Couldn't get context", error);
    
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    clError("Couldn't create command queue", error);
    
    
    sprintf(options_buffer, "-D DATA_DIM=%d -D IMAGE_HEIGHT=%d -D IMAGE_WIDTH=%d -D TRANSFER_FUNC_SIZE=%d -D ELEMENTS_PER_THREAD_X=%d -D ELEMENTS_PER_THREAD_Y=%d -D USE_TEXTURE_DATA=%d -D USE_TEXTURE_TRANSFER=%d -D USE_TRILINEAR=%d -D INTERLEAVED=%d -D USE_SHARED_TRANSFER=%d -D USE_CONSTANT_TRANSFER=%d -D UNROLL_FACTOR=%d",
            DATA_DIM,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            TRANSFER_FUNC_SIZE,
            (int)pow(2,config[ELEMENTS_PER_THREAD_X]),
            (int)pow(2,config[ELEMENTS_PER_THREAD_Y]),
            config[USE_TEXTURE_DATA],
            config[USE_TEXTURE_TRANSFER],
            USE_TRILINEAR,
            config[INTERLEAVED],
            config[USE_SHARED_TRANSFER],
            config[USE_CONSTANT_TRANSFER],
            (int)pow(2,config[UNROLL_FACTOR])
    );
    
    cl_kernel kernel = buildKernel("raycast.cl", "raycast", options_buffer, context, device, &error);
    if(error != CL_SUCCESS){
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -3.0;
    }
    
    cl_mem data_device;
    if(config[USE_TEXTURE_DATA]){
        cl_image_format image_format_data;
        image_format_data.image_channel_order = CL_R;
        image_format_data.image_channel_data_type = CL_FLOAT;
        
        data_device = clCreateImage3D(context,
                                      CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                      &image_format_data,
                                      DATA_DIM,
                                      DATA_DIM,
                                      DATA_DIM,
                                      DATA_DIM*sizeof(float),
                                      DATA_DIM*DATA_DIM*sizeof(float),
                                      data_host,
                                      &error);
        
        
    }
    else{
        data_device = clCreateBuffer(context,
                                     CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,
                                     sizeof(float)*DATA_DIM*DATA_DIM*DATA_DIM,
                                     data_host,
                                     &error);
    }
    
    
    cl_mem image_device = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*IMAGE_WIDTH*IMAGE_HEIGHT, NULL, &error);
    
    cl_mem transfer_device;
    if(config[USE_TEXTURE_TRANSFER]){
        cl_image_format image_format_transfer;
        image_format_transfer.image_channel_order = CL_RGBA;
        image_format_transfer.image_channel_data_type = CL_FLOAT;
        
        transfer_device = clCreateImage2D(context,
                                          CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                          &image_format_transfer,
                                          TRANSFER_FUNC_SIZE,
                                          1,
                                          TRANSFER_FUNC_SIZE*sizeof(cl_float4),
                                          transfer_host,
                                          &error);
    }
    else{
        
        if(config[USE_CONSTANT_TRANSFER]){
            transfer_device = clCreateBuffer(context,
                                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                             sizeof(cl_float4)*TRANSFER_FUNC_SIZE,
                                             transfer_host,
                                             &error);
        }
        else{
            transfer_device = clCreateBuffer(context,
                                             CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                             sizeof(cl_float4)*TRANSFER_FUNC_SIZE,
                                             transfer_host,
                                             &error);
        }
        
    }
    
    
    
    clError("Error allocating memory",error);
    
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device);
    clError("Error setting kernel argument 0",error);
    
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &image_device);
    clError("Error setting kernel argument 1",error);
    
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &transfer_device);
    clError("Error setting kernel argument 2",error);
    
    
    
    
    double time = 0.0;    
    
    if(invalid_work_group_size(device, kernel, 2, local_work_size, global_work_size)){
        time = -1.0;
    }
    else{
        cl_event event;
        //printf("%zu %zu %zu %zu\n", global_work_size[0], global_work_size[1], local_work_size[0], local_work_size[1]);
        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
        if(error != CL_SUCCESS){
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            clReleaseMemObject(data_device);
            clReleaseMemObject(image_device);
            clReleaseMemObject(transfer_device);
            //clWaitForEvents(1, &event);
            //clReleaseEvent(event);
            return -4.0;
        }
        
        
        error = clFinish(queue);
        clError("Error waiting for kernel",error);
        if(error != CL_SUCCESS){
            time = -1.0;
        }
        else{
            cl_ulong start_time, end_time;
            error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
            error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
            time = (double)((end_time-start_time)/1000.0);
            clError("Error timing",error);    
            
            clEnqueueReadBuffer(queue,
                                image_device,
                                CL_TRUE,
                                0,
                                sizeof(int)*IMAGE_WIDTH*IMAGE_HEIGHT,
                                image_host,
                                0,
                                NULL,
                                NULL);
            clError("Error reading stuff", error);
        }
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    
    
    
    clReleaseMemObject(data_device);
    clReleaseMemObject(image_device);
    clReleaseMemObject(transfer_device);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return time;
}

void run_on_configurations(int* configurations, int n_run_configurations, int n_total_configurations, float* data_host, cl_float4* transfer_host, int* correct_image, char** argv){
    
    
    
    int* image_host = (int*)calloc(sizeof(int),IMAGE_WIDTH*IMAGE_HEIGHT);
    cl_device_id device = get_selected_device();
    
    print_comment(device, argv);
    
    int i = get_start_iteration();
    int j = 0;
    while(i < n_total_configurations && j < n_run_configurations){
        int* temp_config = get_config_for_number(configurations[i], param_limits, n_parameters);
        
        
        fprintf(stderr, "%d\t", i);
        for(int p = 0; p < n_parameters; p++){
            fprintf(stderr, "%d ", temp_config[p]);
        }
        fprintf(stderr, "%s\n", timestamp());
        
        
        double time = raycast_ocl(data_host,
                                  transfer_host,
                                  image_host,
                                  device,
                                  temp_config);
        
        
        
        if(time > 0){
            if(!check_image(image_host, correct_image)){
                time = -2.0;
            }
        }        
        
        
        for(int p = 0; p < n_parameters; p++){
            printf("%d ", temp_config[p]);
        }
        //printf("\n");
        printf("%f\n", time);
        
        if(ignore_crashes_when_counting()){
            j++;
        }
        else{
            if(time > 0){
                j++;
            }
        }
        
        i++;
        free(temp_config);

        if(get_use_time_threshold() && time > get_time_threshold() && j >= get_min_second_stage()){
            break;
        }
        if(get_use_time_threshold() && j >= get_max_second_stage()){
            break;
        }
    }
}

          
int main(int argc, char** argv){
    
    parse_args(argc, argv);
    
    float* data_host = create_data();
    
    int* correct_image = NULL;
    if(get_correct_file() != NULL){
        correct_image = load_correct(get_correct_file(), IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    else{
        printf("#WARNING: No correct file provided, output check will not be performed\n");
    }
    cl_float4* transfer_host = create_transfer();
    
    int n_run_configurations;
    int n_total_configurations;
    int* configurations = create_configurations(param_limits, n_parameters, argc, argv, &n_run_configurations, &n_total_configurations);
    
    
                        
    if(perform_self_test()){
        int* image_host = (int*)malloc(sizeof(int)*IMAGE_WIDTH*IMAGE_HEIGHT);
        cl_device_id device = get_selected_device();
        print_comment(device, argv);
        double time = raycast_ocl(data_host, transfer_host, image_host, device, global_config);
        if(check_image(image_host, correct_image))
            printf("Self test sucessfull, time: %f\n", time);
        if(get_output_file() != NULL){
            printf("Writing output to %s\n", get_output_file());
            write_image_raw(get_output_file(), image_host, IMAGE_WIDTH, IMAGE_HEIGHT);
        }
    }
    else{
        run_on_configurations(configurations,
                              n_run_configurations,
                              n_total_configurations,
                              data_host,
                              transfer_host,
                              correct_image,
                              argv);
    }
    
    //print_image(image_host);
    //write_image_raw("pic.bin",image_host, IMAGE_WIDTH, IMAGE_HEIGHT);
    
}

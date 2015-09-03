// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <CL/cl.h>
#include <math.h>

#include "../common/clutil.h"
#include "../common/configurations.h"
#include "../common/io.h"
#include "../common/parser.h"

// Tuning parameters
int LOCAL_SIZE_X =              0;
int LOCAL_SIZE_Y =              1;
int LOCAL_SIZE_Z =              2;
int ELEMENTS_PER_THREAD_X =     3;
int ELEMENTS_PER_THREAD_Y =     4;
int ELEMENTS_PER_THREAD_Z =     5;
int USE_TEXTURE =               6;
int USE_LOCAL =                 7;
int PRECOMPUTE =                8;
int PRECOMPUTE_DIST =           9;
//int INTERLEAVED =               10;
//int OUTER_LOOP =                11;
//int INNER_LOOP =                12;

int global_config[] = {3,3,2,0,0,0,0,0,0,0};
int param_limits[] =  {6,6,6,6,6,6,2,2,2,2}; //Or, rather, the limit + 1
int n_parameters = 10;

//Problem parameters
const int IMAGE_WIDTH = 128;
const int IMAGE_HEIGHT = 64;
const int IMAGE_DEPTH = 64;
const int PADDING = 2;
const int FILTER_WIDTH = 5;
const int FILTER_HEIGHT = 5;
const int FILTER_DEPTH = 3;


int index(int x, int y, int z){
    int width = IMAGE_WIDTH+2*PADDING;
    int height = IMAGE_HEIGHT+2*PADDING;
    int z_base = width*height*PADDING;
    int y_base = width*PADDING;
    int x_base = PADDING;
    
    
    return z_base + z*height*width + y_base +  y*width + x_base + x;
}

float* create_distance_filter(){
    float* filter = (float*)malloc(sizeof(float)*FILTER_WIDTH*FILTER_HEIGHT*FILTER_DEPTH);

    for(int i= 0; i < FILTER_HEIGHT; i++){
        for(int j = 0; j < FILTER_WIDTH; j++){
            for(int k = 0; k < FILTER_DEPTH; k++){
                int index = k*FILTER_WIDTH*FILTER_HEIGHT + i*FILTER_WIDTH+j;
                
                int di = i - FILTER_HEIGHT/2;
                int dj = j - FILTER_WIDTH/2;
                int dk = k - FILTER_DEPTH/2;
                
                float d = sqrt(di*di + dj*dj + dk*dk);
                
                filter[index] = exp(-(d*d)/(2*4));
            }
        }
    }

    return filter;
}

float* create_color_filter(){
    
    float* filter = (float*)malloc(sizeof(float)*256);
    
    
    for(int i = 0; i < 256; i++){
        filter[i] = exp(-(i*i)/(2*85));
    }
    
    return filter;
}


unsigned char* copy_to_padded(unsigned char* input){
    unsigned char* padded_input = (unsigned char*)calloc(sizeof(unsigned char), (IMAGE_WIDTH+2*PADDING) * (IMAGE_HEIGHT+2*PADDING) * (IMAGE_DEPTH + 2*PADDING));
    
    for(int z = 0; z < IMAGE_DEPTH; z++){
        for(int y = 0; y < IMAGE_HEIGHT; y++){
            for(int x = 0; x < IMAGE_WIDTH; x++){
                int padded_index = index(x,y,z);
                int plain_index = z*IMAGE_WIDTH*IMAGE_HEIGHT + y * IMAGE_WIDTH + x;
                
                padded_input[padded_index] = input[plain_index];
            }
        }
    }
    return padded_input;
}

void copy_from_padded(unsigned char* output, unsigned char* padded_output){
    for(int z = 0; z < IMAGE_DEPTH; z++){
        for(int y = 0; y < IMAGE_HEIGHT; y++){
            for(int x = 0; x < IMAGE_WIDTH; x++){
                int padded_index = index(x,y,z);
                int plain_index = z*IMAGE_WIDTH*IMAGE_HEIGHT + y * IMAGE_WIDTH + x;
                
                
                
                output[plain_index] = padded_output[padded_index];
            }
        }
    }
}

unsigned char* create_input(){
    unsigned char* input = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_WIDTH*IMAGE_DEPTH*IMAGE_HEIGHT);
    
    for(int y = 0; y < IMAGE_HEIGHT; y++){
        for(int x = 0; x < IMAGE_WIDTH; x++){
            for(int z = 0; z < IMAGE_DEPTH; z++){
                int index = z*IMAGE_HEIGHT*IMAGE_WIDTH + y*IMAGE_WIDTH + x;
                
                
                input[index] = (x/10)*20 + (y/15)*20 + (z/20)*10;
                
                input[index] += rand() % 20;
            }
        }
    }
    return input;
}



void bilateral_cpu(unsigned char* input, unsigned char* output){
    
    float* dist_filter = create_distance_filter();
    float* color_filter = create_color_filter();
    
    for(int x = 0; x < IMAGE_WIDTH; x++){
        for(int y = 0; y < IMAGE_HEIGHT; y++){
            for(int z = 0; z < IMAGE_DEPTH; z++){
                
                float sum = 0.0;
                int ci = index(x,y,z);
                float norm = 0.0;
                
                for(int fz = 0; fz < FILTER_DEPTH; fz++){
                    for(int fy = 0; fy < FILTER_HEIGHT; fy++){
                        for(int fx = 0; fx < FILTER_WIDTH; fx++){
                            int i = index(x + fx - FILTER_WIDTH/2, y + fy - FILTER_HEIGHT/2, z + fz - FILTER_DEPTH/2);
                            int fi = fz*FILTER_WIDTH*FILTER_HEIGHT + fy * FILTER_WIDTH + fx;
                            
                            int color_diff = abs(input[ci] - input[i]);
                            
                            
                            norm += dist_filter[fi]*color_filter[color_diff];
                            sum += input[i]*dist_filter[fi]*color_filter[color_diff];
                        }       
                    }
                }
                
                output[index(x,y,z)] = sum/norm;
            }
        }
    }
}

void print2d(float* buffer, int width, int height){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            printf("%03.2f ", buffer[i*width+j]);
        }
        printf("\n");
    }
    printf("\n");
}

double bilateral_ocl(unsigned char* input, unsigned char* output, cl_device_id device, int* config){
    
    int lwsx = pow(2, config[LOCAL_SIZE_X]);
    int lwsy = pow(2, config[LOCAL_SIZE_Y]);
    int lwsz = pow(2, config[LOCAL_SIZE_Z]);
    int eptx = pow(2, config[ELEMENTS_PER_THREAD_X]);
    int epty = pow(2, config[ELEMENTS_PER_THREAD_Y]);
    int eptz = pow(2, config[ELEMENTS_PER_THREAD_Z]);
    const size_t local_work_size[3] = {lwsx,lwsy,lwsz};
    const size_t global_work_size[3] = {(IMAGE_WIDTH/eptx),IMAGE_HEIGHT/epty, IMAGE_DEPTH/eptz};
    
    if(invalid_work_group_size_static(device, 3, local_work_size, global_work_size)){
        return -1;
    }
    
    cl_int error;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    clError("Couldn't get context", error);
    
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    clError("Couldn't create command queue", error);
    
    char* kernelName = "bilateral.cl";
    char options_buffer [400];
    sprintf(options_buffer, "-D ELEMENTS_PER_THREAD_X=%d -D ELEMENTS_PER_THREAD_Y=%d -D ELEMENTS_PER_THREAD_Z=%d"
    " -D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D LOCAL_SIZE_Z=%d"
    " -D FILTER_WIDTH=%d -D FILTER_HEIGHT=%d -D FILTER_DEPTH=%d"
    " -D IMAGE_WIDTH=%d -D IMAGE_HEIGHT=%d -D IMAGE_DEPTH=%d"
    " -D PADDING=%d -D USE_TEXTURE=%d -D USE_LOCAL=%d -D PRECOMPUTE=%d -D PRECOMPUTE_DIST=%d -D INTERLEAVED=%d"
    " -D INNER_LOOP=%d -D OUTER_LOOP=%d",
    (int)pow(2,config[ELEMENTS_PER_THREAD_X]),
            (int)pow(2,config[ELEMENTS_PER_THREAD_Y]),
            (int)pow(2,config[ELEMENTS_PER_THREAD_Z]),
            (int)pow(2,config[LOCAL_SIZE_X]),
            (int)pow(2,config[LOCAL_SIZE_Y]),
            (int)pow(2,config[LOCAL_SIZE_Z]),
            FILTER_WIDTH,
            FILTER_HEIGHT,
            FILTER_DEPTH,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            IMAGE_DEPTH,
            PADDING,
            config[USE_TEXTURE],
            config[USE_LOCAL],
            config[PRECOMPUTE],
            config[PRECOMPUTE_DIST],
            0,
            0,
            0
    );
    
    kernel = buildKernel(kernelName, "bilateral", options_buffer, context, device, &error);
    if(error != CL_SUCCESS){
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -3.0;
    }
    
    
    if(invalid_work_group_size(device, kernel, 3, local_work_size, global_work_size)){
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return  -1.0;
    }
    
    
    int size = (IMAGE_WIDTH+2*PADDING)*(IMAGE_HEIGHT+2*PADDING)*(IMAGE_DEPTH+2*PADDING);
    
    cl_mem input_device;
    if(config[USE_TEXTURE]){
        cl_image_format image_format;
        image_format.image_channel_order = CL_R;
        image_format.image_channel_data_type = CL_UNSIGNED_INT8;
        input_device = clCreateImage3D(context,
                                       CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                       &image_format,
                                       IMAGE_WIDTH+2*PADDING,
                                       IMAGE_HEIGHT+2*PADDING,
                                       IMAGE_DEPTH+2*PADDING,
                                       IMAGE_WIDTH+2*PADDING*sizeof(unsigned char),
                                       (IMAGE_WIDTH+2*PADDING)*(IMAGE_HEIGHT+2*PADDING)*sizeof(unsigned char),
                                       input,
                                       &error);
    }else{
        input_device = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,  size, input, &error);
    }
    
    cl_mem output_device = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error);
    
    float* dist_filter = create_distance_filter();
    float* color_filter = create_color_filter();
    
    cl_mem dist_filter_device;
    if(config[PRECOMPUTE_DIST]){
        dist_filter_device = clCreateBuffer(context,
                                            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                            FILTER_HEIGHT*FILTER_WIDTH*FILTER_DEPTH*sizeof(float),
                                            dist_filter, &error);
    }
    cl_mem color_filter_device;
    if(config[PRECOMPUTE]){
        color_filter_device = clCreateBuffer(context,
                                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                             256*sizeof(float), color_filter, &error);
    }
    
    clError("Error allocating memory",error);
    
    
    int arg = 0;
    error = clSetKernelArg(kernel, arg, sizeof(cl_mem), &input_device);
    clError("Error setting kernel argument", error);
    arg++;
    
    
    if(config[PRECOMPUTE]){
        error = clSetKernelArg(kernel, arg, sizeof(cl_mem), &color_filter_device);
        clError("Error setting kernel argument", error);
        arg++;
    }
        
    if(config[PRECOMPUTE_DIST]){
        error = clSetKernelArg(kernel, arg, sizeof(cl_mem), &dist_filter_device);
        clError("Error setting kernel argument", error);
        arg++;
    }
    
    error = clSetKernelArg(kernel, arg, sizeof(cl_mem), &output_device);
    clError("Error setting kernel argument", error);
    arg++;
    
    
    cl_event event;
    double time;
    error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);
    clError("enqueue kernel", error);
    if(error != CL_SUCCESS){
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        if(config[PRECOMPUTE_DIST]){
            clReleaseMemObject(dist_filter_device);
        }
        if(config[PRECOMPUTE]){
            clReleaseMemObject(color_filter_device);
        }
        clReleaseMemObject(input_device);
        clReleaseMemObject(output_device);
        //clWaitForEvents(1, &event);
        //clReleaseEvent(event);
        return -4.0;
    }
    
    error = clFinish(queue);
    clError("Error waiting for kernel",error);
    if(error != CL_SUCCESS){
        return -1.0;
    }
    else{
        cl_ulong start_time, end_time;
        error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
        time = (double)(end_time-start_time)/1000.0;
        clError("Error timing",error);
        
        
        
        error = clEnqueueReadBuffer(queue,
                                    output_device,
                                    CL_TRUE,
                                    0,
                                    size,
                                    output,
                                    0,
                                    NULL,
                                    NULL);
        clError("Error reading stuff", error);
    }
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    
    clReleaseMemObject(input_device);
    clReleaseMemObject(output_device);
    if(config[PRECOMPUTE_DIST]){
        clReleaseMemObject(dist_filter_device);
    }
    if(config[PRECOMPUTE]){
        clReleaseMemObject(color_filter_device);
    }
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return time;
}



char* timestamp(){
    time_t ltime;
    ltime=time(NULL);
    char* ts = malloc(50);
    sprintf(ts, "%s",asctime( localtime(&ltime) ) );
    return ts;
}

int compare(unsigned char* a, unsigned char* b, int length){
    
    if(a == NULL || b == NULL){
        return 1;
    }
    
    int n_errors = 0;
    for(int i = 0; i < length; i++){
        float diff = abs((int)a[i] - (int)b[i]);
        if(diff > 3){
            fprintf(stderr,"Error at: %d: %d %d\n", i, a[i], b[i]);
            n_errors++;
        }
        if(n_errors > 10){
            break;
        }
    }
    return n_errors == 0;
}

void print_comment(cl_device_id device, char** argv){
    printf("# %s\n", argv[0]);

    time_t ltime;
    ltime=time(NULL);
    printf("# %s",asctime( localtime(&ltime) ) );

    char name[100];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 100, name, NULL);
    printf("# %s\n", name);
    printf("\n");

    printf("# IMAGE_HEIGHT %d\n", IMAGE_HEIGHT);
    printf("# IMAGE_WIDTH %d\n", IMAGE_WIDTH);
    printf("# IMAGE_DEPTH %d\n", IMAGE_DEPTH);
    printf("# PADDING %d\n", PADDING);
    printf("# FILTER_WIDTH %d\n", FILTER_WIDTH);
    printf("# FILTER_HEIGHT %d\n", FILTER_HEIGHT);
    printf("# FILTER_DEPTH %d\n", FILTER_DEPTH);
    printf("\n");
}

void run_on_configurations(int* configurations,
                           int n_run_configurations,
                           int n_total_configurations,
                           unsigned char* padded_input,
                           unsigned char* padded_correct_output,
                           char** argv){
    
    unsigned char* padded_output = (unsigned char*)malloc(sizeof(unsigned char)*(IMAGE_WIDTH+(2*PADDING))*(IMAGE_HEIGHT+(2*PADDING))*(IMAGE_DEPTH+2*PADDING));
    unsigned char* output = (unsigned char*)calloc(sizeof(unsigned char),IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH);
    unsigned char* correct_output = NULL;
    if(padded_correct_output){
        correct_output = (unsigned char*)calloc(sizeof(unsigned char),IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH);
        copy_from_padded(correct_output, padded_correct_output);
    }

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
        


        double time = bilateral_ocl(padded_input, padded_output, device, temp_config);
        copy_from_padded(output, padded_output);


        if(time > 0 && correct_output){
            if(!compare(output, correct_output,IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_DEPTH)){
                time = -2.0;
            }
        }


        for(int p = 0; p < n_parameters; p++){
            printf("%d ", temp_config[p]);
        }
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
    }
}


int main(int argc, char** argv){
    
    parse_args(argc, argv);
    
    unsigned char* input = create_input();
    unsigned char* padded_input = copy_to_padded(input);
    
    unsigned char* output = malloc(sizeof(unsigned char)*IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH);
    unsigned char* padded_output = malloc(sizeof(unsigned char)*(IMAGE_WIDTH+2*PADDING)*(IMAGE_HEIGHT+2*PADDING)*(IMAGE_DEPTH+2*PADDING));
    
    
    
    unsigned char* output_gold = NULL;
    unsigned char* padded_output_gold= NULL;
    if(get_correct_file() != NULL){
        output_gold = load_raw_buffer(get_correct_file(), IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH);
        padded_output_gold = copy_to_padded(output_gold);
    }
    else{
        printf("#Warning: No correct file provided, output check will not be performed\n");
    }
    
    int n_run_configurations;
    int n_total_configurations;
    int* configurations = create_configurations(param_limits, n_parameters, argc, argv, &n_run_configurations, &n_total_configurations);
    
    
    if(perform_self_test()){
        cl_device_id device = get_selected_device();
        
        print_comment(device, argv);
        
        double time = bilateral_ocl(padded_input, padded_output, device, global_config);
        copy_from_padded(output, padded_output);
        //write_ppm_uchar(output, IMAGE_WIDTH, IMAGE_HEIGHT);
        
        
        if(compare(output, output_gold, (IMAGE_WIDTH)*(IMAGE_HEIGHT)*IMAGE_DEPTH))
            printf("Self test successfull, time: %f\n", time);
        if(get_output_file() != NULL){
            printf("Writing output to %s\n", get_output_file());
            write_raw_buffer(get_output_file(), output, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH);
        }
        
    }
    else{
        
        run_on_configurations(configurations,
                              n_run_configurations,
                              n_total_configurations,
                              padded_input,
                              padded_output_gold,
                              argv);
    }
}


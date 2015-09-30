// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#include <math.h>
#include "../common/clutil.h"
#include "../common/configurations.h"
#include "../common/io.h"
#include "../common/parser.h"

int config[] = {3,3,0,0,0,0,0,0,0,0};
int limits[] = {8,8,8,8,2,2,4,3,3,2,2};
int n_parameters = 11;

#define LOCAL_SIZE_X            0
#define LOCAL_SIZE_Y            1
#define ELEMENTS_PER_THREAD_X   2
#define ELEMENTS_PER_THREAD_Y   3
#define USE_TEXTURE_LEFT        4
#define USE_TEXTURE_RIGHT       5
#define UNROLL_DISPARITY_LOOP_FACTOR   6
#define UNROLL_RADIUS_X_FACTOR	7
#define UNROLL_RADIUS_Y_FACTOR	8
#define USE_LOCAL_LEFT          9
#define USE_LOCAL_RIGHT         10

const int MIN_DISPARITY = -8;
const int MAX_DISPARITY = 8;
const int RADIUS = 2;
const int IMAGE_WIDTH = 256;
const int IMAGE_HEIGHT = 256;


char* timestamp(){
    time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
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
    
    printf("# MIN_DISPARITY: %d\n", MIN_DISPARITY);
    printf("# MAX_DISPARITY %d\n", MAX_DISPARITY);
    printf("# RADIUS %d\n", RADIUS);
    printf("# IMAGE_WIDTH %d\n", IMAGE_WIDTH);
    printf("# IMAGE_HEIGHT %d\n", IMAGE_HEIGHT);
    
    printf("\n");
}


int check_image(int* image, int* correct, int width, int height){
    if(image == NULL || correct == NULL){
        return 1;
    }
    
    int n_errors = 0;
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            
            if(image[i*width + j] != correct[i*width + j]){
                n_errors += 1;
                if(n_errors < 10){
                    fprintf(stderr, "Error at %d,%d, expected %d, found %d\n", i, j, correct[i*width+j], image[i*width+j]);
                }
            }
        }
    }
    if(n_errors > 0){
        fprintf(stderr,"%d errors in total\n", n_errors);
    }
    
    return n_errors == 0;
}


int* compute_disparity_cpu(int* left_image, int* right_image, int width, int height, int min_disparity, int max_disparity, int radius){
    
    int* disparity = (int*)malloc(sizeof(int)* width * height);
        
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            
            
            int min = 999999;
            int min_d = 0;
            for(int d = min_disparity; d <= max_disparity; d++){
                
                int sum = 0;
                for(int i = -radius; i <= radius; i++){
                    for(int j = -radius; j <= radius; j++){
                        
                        int xxd = x + i + d;
                        int xx = x + i;
                        int yy = y + j;
                        
                        if( xx >= width) xx = width-1;
                        if( xx < 0) xx = 0;
                        
                        if( xxd >= width) xxd = width-1;
                        if( xxd < 0) xxd = 0;
                        
                        if( yy >= height) yy = height-1;
                        if( yy < 0) yy = 0;
                        
                        int index1 = yy*width + xx;
                        int index2 = yy*width + xxd;
                
                
                                              
                        unsigned char* left_pixel = (unsigned char *)&left_image[index1];
                        unsigned char* right_pixel = (unsigned char *)&right_image[index2];
                        int absdiff = 0;
                        for (int k=0; k<4; k++){
                            absdiff += abs((int)(left_pixel[k] - right_pixel[k]));
                        }
                        sum += absdiff;
                    }
                }
                
                if(sum < min){
                    min = sum;
                    min_d = d;
                }
            }
            disparity[y*width + x] = ((min_d+max_disparity)*5);
        }
    }
    
    return disparity;
}


double compute_disparity_ocl(int* left_image, int* right_image, int* disparity, int width, int height, int min_disparity, int max_disparity, int radius, cl_device_id device, int* temp_config){
    
    int lwsx = pow(2, temp_config[LOCAL_SIZE_X]);
    int lwsy = pow(2, temp_config[LOCAL_SIZE_Y]);
    int eptx = pow(2, temp_config[ELEMENTS_PER_THREAD_X]);
    int epty = pow(2, temp_config[ELEMENTS_PER_THREAD_Y]);
    const size_t local_work_size[2] = {lwsx,lwsy};
    const size_t global_work_size[2] = {(width/eptx),height/epty};
    
    if(invalid_work_group_size_static(device, 2, local_work_size, global_work_size)){
        return -1;
    }
    
    cl_int error;
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    clError("Couldn't get context", error);
    
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    clError("Couldn't create command queue", error);
    
    char options_buffer[400];
    sprintf(options_buffer, "-D IMAGE_HEIGHT=%d -D IMAGE_WIDTH=%d -D MIN_DISPARITY=%d"
    " -D MAX_DISPARITY=%d -D RADIUS=%d -D ELEMENTS_PER_THREAD_X=%d"
    " -D ELEMENTS_PER_THREAD_Y=%d -D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d"
    " -D USE_TEXTURE_LEFT=%d -D USE_TEXTURE_RIGHT=%d -D UNROLL_DISPARITY_LOOP_FACTOR=%d"
    " -D USE_LOCAL_LEFT=%d -D USE_LOCAL_RIGHT=%d -D UNROLL_RADIUS_X_FACTOR=%d"
    " -D UNROLL_RADIUS_Y_FACTOR=%d",
    height,
    width,
    min_disparity,
    max_disparity,
    radius,
    (int)pow(2,temp_config[ELEMENTS_PER_THREAD_X]),
            (int)pow(2,temp_config[ELEMENTS_PER_THREAD_Y]),
            (int)pow(2,temp_config[LOCAL_SIZE_X]),
            (int)pow(2,temp_config[LOCAL_SIZE_Y]),
            temp_config[USE_TEXTURE_LEFT],
            temp_config[USE_TEXTURE_RIGHT],
            (int)pow(2,temp_config[UNROLL_DISPARITY_LOOP_FACTOR]),
            temp_config[USE_LOCAL_LEFT],
            temp_config[USE_LOCAL_RIGHT],
            (int)pow(2,temp_config[UNROLL_RADIUS_X_FACTOR]),
            (int)pow(2,temp_config[UNROLL_RADIUS_Y_FACTOR])
    );
    cl_kernel kernel = buildKernel("stereo.cl", "stereo", options_buffer, context, device, &error);
    
    if(error != CL_SUCCESS){
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -3.0;
    }
    
    
    
    cl_mem left_image_device;
    if(temp_config[USE_TEXTURE_LEFT]){
        cl_image_format image_format_left;
        image_format_left.image_channel_order = CL_R;
        image_format_left.image_channel_data_type = CL_SIGNED_INT32;
        
        left_image_device = clCreateImage2D(context,
                                            CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                            &image_format_left,
                                            width,
                                            height,
                                            width*sizeof(int),
                                            left_image,
                                            &error);
    }
    else{
        left_image_device = clCreateBuffer(context,
                                           CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,
                                           sizeof(int)*width*height,
                                           left_image,
                                           &error);
    }
    clError("Couldn't allocate left memory", error);
    
    cl_mem right_image_device;
    if(temp_config[USE_TEXTURE_RIGHT]){
        cl_image_format image_format_right;
        image_format_right.image_channel_order = CL_R;
        image_format_right.image_channel_data_type = CL_SIGNED_INT32;
        
        right_image_device = clCreateImage2D(context,
                                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                             &image_format_right,
                                             width,
                                             height,
                                             width*sizeof(int),
                                             right_image,
                                             &error);
    }
    else{
        
        right_image_device = clCreateBuffer(context,
                                            CL_MEM_COPY_HOST_PTR|CL_MEM_READ_WRITE,
                                            sizeof(int)*width*height,
                                            right_image,
                                            &error);
    }
    clError("Couldn't allocate right memory", error);
    
    cl_mem disparity_device = clCreateBuffer(context,
                                             CL_MEM_READ_WRITE,
                                             sizeof(int)*width*height,
                                             NULL,
                                             &error);
    clError("Couldn't allocate disp memory", error);
    
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &left_image_device);
    clError("Error setting kernel argument 0",error);
    
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &right_image_device);
    clError("Error setting kernel argument 1",error);
    
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &disparity_device);
    clError("Error setting kernel argument 2",error);
    
    
    double time;
    if(invalid_work_group_size(device, kernel, 2, local_work_size, global_work_size)){
        
        time = -1.0;
    }
    else{
        //printf("%zu %zu, %zu, %zu\n", global_work_size[0], global_work_size[1], local_work_size[0], local_work_size[1]);
        cl_event event;
        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
        if(error != CL_SUCCESS){
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            clReleaseMemObject(left_image_device);
            clReleaseMemObject(right_image_device);
            clReleaseMemObject(disparity_device);
            //clWaitForEvents(1, &event);
            //clReleaseEvent(event);
            return -4.0;
        }
        
        error = clFinish(queue);
        clError("Error waiting for kernel",error);
        
        cl_ulong start_time, end_time;
        error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
        time = (double)((end_time-start_time)/1000.0);
        clError("Error timing",error); 
        
        
        error = clEnqueueReadBuffer(queue,
                                    disparity_device,
                                    CL_TRUE,
                                    0,
                                    sizeof(int)*width*height,
                                    disparity,
                                    0,
                                    NULL,
                                    NULL);
        clError("Error reading stuff", error);
        
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    
    clReleaseMemObject(left_image_device);
    clReleaseMemObject(right_image_device);
    clReleaseMemObject(disparity_device);
    
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return time;
}

int check_config(int* temp_config){
    int r = (LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS)*USE_LOCAL_RIGHT;
    int l = (LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS)*USE_LOCAL_LEFT;
    
    printf("%x\n", (unsigned int)(r+l*sizeof(int)));
    if((r + l)*sizeof(int) > 0xc000){
        return 0;
    }
    return 1;
}


void run_on_configurations(int* configurations, int n_total_configurations, int n_run_configurations, int* left_image, int* right_image, int* disparity_correct, int height, int width, char** argv){

    int* disparity = (int*)malloc(sizeof(int)*width*height);
    cl_device_id device = get_selected_device();

    print_comment(device, argv);
        
    int i = get_start_iteration();
    int j = 0;
    while(i < n_total_configurations && j < n_run_configurations){
        int* temp_config = get_config_for_number(configurations[i], limits, n_parameters);
        
        fprintf(stderr, "%d\t", i);
        for(int p = 0; p < n_parameters; p++){
            fprintf(stderr, "%d ", temp_config[p]);
        }
        fprintf(stderr, "%s\n", timestamp());
        
        double time = compute_disparity_ocl(left_image,
                                            right_image,
                                            disparity,
                                            width,
                                            height,
                                            MIN_DISPARITY,
                                            MAX_DISPARITY,
                                            RADIUS,
                                            device,
                                            temp_config
                                           );
        

        if(time > 0){
            if(!check_image(disparity, disparity_correct, width, height)){
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

        if(get_use_time_threshold() && time > get_time_threshold() && i >= get_min_second_stage()){
            break;
        }
        if(get_use_time_threshold() && i >= get_max_second_stage()){
            break;
        }
    }
}


int* generate_test_pattern(int width, int height, float offset){
    int* image = (int*)malloc(sizeof(int)*width*height);
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            
            int x = (j + offset*(sin(i*0.05)+cos(j*0.05)));
            int y = i;
            
            unsigned char r = (1+sin(x*0.1))*127;
            unsigned char g = (1+cos((x+y)*0.05))*127;
            unsigned char b = (2 + sin(x*0.04) + cos(y*0.04))*128;
            unsigned char a = 0;
            
            image[i*width + j] = (((unsigned int)a<<24) | ((unsigned int)b<<16) | ((unsigned int)g<<8) | ((unsigned int)r));
        }
    }
    return image;
}


int main(int argc, char** argv){

    //int height, width;
    parse_args(argc, argv);
    
    int* left_image = generate_test_pattern(IMAGE_WIDTH, IMAGE_HEIGHT, 10);
    int* right_image = generate_test_pattern(IMAGE_WIDTH, IMAGE_HEIGHT, 0);
    
    //int* disparity_correct = compute_disparity_cpu(left_image, right_image, IMAGE_WIDTH, IMAGE_HEIGHT, MIN_DISPARITY, MAX_DISPARITY, RADIUS);
    int* disparity_correct = NULL;
    if(get_correct_file() != NULL){
        disparity_correct = load_correct(get_correct_file(), IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    else{
        printf("#Warning: No correct file provided, output check will not be performed\n");
    }
    
    //write_image_raw("pic.bin", disparity_correct, IMAGE_WIDTH, IMAGE_HEIGHT);
    //write_ppm_bw(disparity_correct, IMAGE_WIDTH, IMAGE_HEIGHT);
    
    int n_run_configurations;
    int n_total_configurations;
    int* configurations = create_configurations(limits, n_parameters, argc, argv, &n_run_configurations, &n_total_configurations);


   
    if(perform_self_test()){
        cl_device_id device = get_selected_device();
        print_comment(device, argv);
        int* disparity = (int*)malloc(sizeof(int)*IMAGE_WIDTH*IMAGE_HEIGHT);
        double time = compute_disparity_ocl(left_image, right_image, disparity, IMAGE_WIDTH, IMAGE_HEIGHT, MIN_DISPARITY, MAX_DISPARITY, RADIUS, device, config);
        if(check_image(disparity, disparity_correct, IMAGE_WIDTH, IMAGE_HEIGHT)){
            printf("Self test successfull, time: %f\n", time);
        }
        if(get_output_file() != NULL){
            printf("Writing output to %s\n", get_output_file());
            write_image_raw(get_output_file(), disparity, IMAGE_WIDTH, IMAGE_HEIGHT);
        }
    }
    else{
        run_on_configurations(configurations,
                              n_total_configurations,
                              n_run_configurations,
                              left_image,
                              right_image,
                              disparity_correct,
                              IMAGE_HEIGHT,
                              IMAGE_WIDTH,
                              argv);
    }
    
    
    //write_ppm_bw(disparity, IMAGE_WIDTH, IMAGE_HEIGHT);
    //write_image_raw("pic.bin", disparity, IMAGE_WIDTH, IMAGE_HEIGHT);
    //write_ppm(left_image, IMAGE_WIDTH, IMAGE_HEIGHT);
}
    

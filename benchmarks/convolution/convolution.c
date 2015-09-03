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
int ELEMENTS_PER_THREAD_X =     2;
int ELEMENTS_PER_THREAD_Y =     3;
int USE_TEXTURE =		4;
int USE_LOCAL =			5;
int FAKE_PADDING =		6;
int INTERLEAVED =		7;
int UNROLL =			8;

int global_config[] = {3,3,1,1,0,0,0,0,0};
int param_limits[] =  {8,8,8,8,2,2,2,2,2}; //Or, rather, the limit + 1
int n_parameters = 9;

//Problem parameters
const int IMAGE_WIDTH = 2048;
const int IMAGE_HEIGHT = 2048;
const int PADDING = 4;
const int FILTER_WIDTH = 5;
const int FILTER_HEIGHT = 5;

float* padded_input_g;
float* filter_g;
cl_device_id device_g;

float* create_filter(int filter_width, int filter_height){
    float* filter = (float*)malloc(sizeof(float)*filter_width*filter_height);

    for(int i= 0; i < filter_height; i++){
        for(int j = 0; j < filter_width; j++){
            filter[i*filter_width+j] = 1;// i*j*0.1;
        }
    }

    return filter;
}


float* copy_to_padded(float* input,  int width, int height, int padding){
	float* padded_input = (float*)calloc(sizeof(float), (width+2*padding) * (height+2*padding));
	for(int i = 0; i < height; i++){
    	for(int j = 0; j < width; j++){
    		padded_input[padding*(width+2*padding) + i*(width+2*padding) + padding+ j] = input[i*width + j];
    	}
	}
	return padded_input;
}

void copy_from_padded(float* output, float* padded_output, int width, int height, int padding){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      output[i*width+j] = padded_output[padding*(width+2*padding) + i*(width+2*padding) + padding+ j];
    }
  }
}

float* create_input(int width, int height){
	float* input = (float*)malloc(sizeof(float)*width*height);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
		  input[i*width + j] = ((i/50)%2)*50 + ((j/70)%2)*20 + (j+i)*0.05;
		}
	}
	return input;
}

int index(int x, int y, int width, int padding){
    return padding*(width+2*padding) + y*(width+2*padding) + x + padding;
}

void convolve_cpu(float* input, float* output, float* filter, int width, int height, int padding){

  for(int w = 0; w < width; w++){
      for(int h = 0; h < height; h++){

          int k = 0;
          float sum = 0.0;
          int fw = floor(FILTER_WIDTH/2);
          int fh = floor(FILTER_HEIGHT/2);
          for(int i = -fw; i <= fw; i++){
              for(int j = -fh; j <= fh; j++){
                  sum += input[index(w+i,h+j,width,padding)] * filter[k++];
              }
          }
          output[index(w,h,width,PADDING)] = sum;
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

double convolve_ocl(float* input, float* output, float* filter, int width, int height, int padding, cl_device_id device, int* config){
    
    int lwsx = pow(2, config[LOCAL_SIZE_X]);
    int lwsy = pow(2, config[LOCAL_SIZE_Y]);
    int eptx = pow(2, config[ELEMENTS_PER_THREAD_X]);
    int epty = pow(2, config[ELEMENTS_PER_THREAD_Y]);
    const size_t local_work_size[2] = {lwsx,lwsy};
    const size_t global_work_size[2] = {(IMAGE_WIDTH/eptx),IMAGE_HEIGHT/epty};
    
    if(invalid_work_group_size_static(device, 2, local_work_size, global_work_size)){
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
    
    char* kernelName = "convolution.cl";
    char options_buffer [300];
    sprintf(options_buffer, "-D ELEMENTS_PER_THREAD_X=%d -D ELEMENTS_PER_THREAD_Y=%d"
    " -D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D FILTER_WIDTH=%d -D FILTER_HEIGHT=%d"
    " -D USE_TEXTURE=%d -D USE_LOCAL=%d -D PADDING=%d -D FAKE_PADDING=%d"
    " -D INTERLEAVED=%d -D UNROLL=%d",
    (int)pow(2,config[ELEMENTS_PER_THREAD_X]),
            (int)pow(2,config[ELEMENTS_PER_THREAD_Y]),
            (int)pow(2,config[LOCAL_SIZE_X]),
            (int)pow(2,config[LOCAL_SIZE_Y]),
            FILTER_WIDTH,
            FILTER_HEIGHT,
            config[USE_TEXTURE],
            config[USE_LOCAL],
            PADDING,
            config[FAKE_PADDING],
            config[INTERLEAVED],
            config[UNROLL]
    );
    
    kernel = buildKernel(kernelName, "convolve", options_buffer, context, device, &error);
    if(error != CL_SUCCESS){
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -3.0;
    }
    
    
    cl_mem filter_device = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                          FILTER_HEIGHT*FILTER_WIDTH*sizeof(float),
                                          filter,
                                          &error);
    
    int align = 4096/8;
    int pitch = ((width+2*PADDING)*sizeof(float)/align);
    pitch += ((width+2*PADDING)*sizeof(float)%align) == 0 ? 0 : 1;
    pitch *= align;
    int pitched_size = pitch*(height+2*PADDING);
    size_t buffer_origin[3] = {0,0,0};
    size_t host_origin[3] = {0,0,0};
    size_t region[3] = {(width+2*PADDING)*sizeof(float), (height+2*PADDING), 1};
    
    cl_mem input_device;
    if( config[USE_TEXTURE]){
        cl_image_format image_format;
        image_format.image_channel_order = CL_R;
        image_format.image_channel_data_type = CL_FLOAT;
        input_device = clCreateImage2D(context,
                                       CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                       &image_format,
                                       width+2*PADDING,
                                       height+2*PADDING,
                                       (width+2*PADDING)*sizeof(float),
                                       input,
                                       &error);
    }else{
        input_device = clCreateBuffer(context,
                                      CL_MEM_READ_WRITE,
                                      pitched_size*sizeof(float),
                                      NULL,
                                      &error);
        
        error = clEnqueueWriteBufferRect(queue,
                                         input_device,
                                         CL_TRUE,
                                         buffer_origin,
                                         host_origin,
                                         region,
                                         pitch,
                                         0,
                                         0,
                                         0,
                                         input,
                                         0,
                                         NULL,
                                         NULL);
        
        
    }
    cl_mem output_device = clCreateBuffer(context, CL_MEM_READ_WRITE, pitched_size*sizeof(float), NULL, &error);
    clError("Error allocating memory",error);
    
    
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_device);
    clError("Error setting kernel argument 0",error);
    
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_device);
    clError("Error setting kernel argument 1",error);
    
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_device);
    clError("Error setting kernel argument 2",error);
    
    error = clSetKernelArg(kernel, 3, sizeof(cl_int), &height);
    clError("Error setting kernel argument 3",error);
    
    error = clSetKernelArg(kernel, 4, sizeof(cl_int), &width);
    clError("Error setting kernel argument 4",error);
    
    error = clSetKernelArg(kernel, 5, sizeof(cl_int), &padding);
    clError("Error setting kernel argument 5",error);
    
    int pitch_in_floats = pitch/sizeof(float);
    error = clSetKernelArg(kernel, 6, sizeof(cl_int), &pitch_in_floats);
    clError("Error setting kernel argument 6",error);
    
    
    
    cl_event event;
    double time;
    if(invalid_work_group_size(device, kernel, 2, local_work_size, global_work_size)){
        time = -1.0;
    }
    else{
        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
        clError("enqueue kernel", error);
        if(error != CL_SUCCESS){
            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            clReleaseMemObject(filter_device);
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
            
            
            
            error = clEnqueueReadBufferRect(queue,
                                            output_device,
                                            CL_TRUE,
                                            buffer_origin,
                                            host_origin,
                                            region,
                                            pitch,
                                            0,
                                            0,
                                            0,
                                            output,
                                            0,
                                            NULL,
                                            NULL);
            clError("Error reading stuff", error);
        }
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    
    clReleaseMemObject(filter_device);
    clReleaseMemObject(input_device);
    clReleaseMemObject(output_device);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return time;
}



char* timestamp(){
    time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
    char* ts = malloc(50);
    sprintf(ts, "%s",asctime( localtime(&ltime) ) );
    return ts;
}

int compare(float* a, float* b, int length){
    
    if(a == NULL || b == NULL){
        return 1;
    }
    
    int n_errors = 0;
    for(int i = 0; i < length; i++){
        float diff = fabs(a[i] - b[i]);
        float rel_diff = diff/fmax(fabs(a[i]), fabs(b[i]));
        if(rel_diff > 1e-5){
            fprintf(stderr,"Error at: %d: %f %.12f, %f, %f\n", i, diff, rel_diff, a[i], b[i]);
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

    time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
    printf("# %s",asctime( localtime(&ltime) ) );

    char name[100];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 100, name, NULL);
    printf("# %s\n", name);
    printf("\n");

    printf("# IMAGE_HEIGHT %d\n", IMAGE_HEIGHT);
    printf("# IMAGE_WIDTH %d\n", IMAGE_WIDTH);
    printf("# PADDING %d\n", PADDING);
    printf("# FILTER_WIDTH %d\n", FILTER_WIDTH);
    printf("# FILTER_HEIGHT %d\n", FILTER_HEIGHT);
    printf("\n");
}

void run_on_configurations(int* configurations,
                           int n_run_configurations,
                           int n_total_configurations,
                           float* padded_input,
                           float* filter,
                           float* padded_correct_output,
                           char** argv){
    
    float* padded_output = (float*)malloc(sizeof(float)*(IMAGE_WIDTH+(2*PADDING))*(IMAGE_HEIGHT+(2*PADDING)));
    float* output = (float*)calloc(sizeof(float),IMAGE_WIDTH*IMAGE_HEIGHT);
    float* correct_output = NULL;
    if(padded_correct_output){
        correct_output = (float*)calloc(sizeof(float),IMAGE_WIDTH*IMAGE_HEIGHT);
        copy_from_padded(correct_output, padded_correct_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
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
        


        double time = convolve_ocl(padded_input, padded_output, filter,IMAGE_WIDTH, IMAGE_HEIGHT, PADDING, device, temp_config);
        copy_from_padded(output, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);


        if(time > 0 && correct_output){
            if(!compare(output, correct_output,IMAGE_HEIGHT*IMAGE_WIDTH)){
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

    float* filter = create_filter(FILTER_WIDTH, FILTER_HEIGHT);
    
    float* input = create_input(IMAGE_WIDTH, IMAGE_HEIGHT);
    float* padded_input = copy_to_padded(input, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    
    float* output = (float*)malloc(sizeof(float)*IMAGE_WIDTH*IMAGE_HEIGHT);
    float* padded_output = (float*)malloc(sizeof(float)*(IMAGE_WIDTH+2*PADDING)*(IMAGE_HEIGHT+2*PADDING));
    
    /*
    float* output_gold = (float*)malloc(sizeof(float)*IMAGE_WIDTH*IMAGE_HEIGHT);
    float* padded_output_gold = (float*)malloc(sizeof(float)*(IMAGE_WIDTH+2*PADDING)*(IMAGE_HEIGHT+2*PADDING));
    convolve_cpu(padded_input, padded_output_gold, filter, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    */
    float* output_gold = NULL;
    float* padded_output_gold= NULL;
    if(get_correct_file() != NULL){
        output_gold = load_correct_float(get_correct_file(), IMAGE_WIDTH, IMAGE_HEIGHT);
        padded_output_gold = copy_to_padded(output_gold, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    }
    else{
        printf("#Warning: No correct file provided, output check will not be performed\n");
    }
    
    //copy_from_padded(output_gold, padded_output_gold, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    
    
    
    int n_run_configurations;
    int n_total_configurations;
    int* configurations = create_configurations(param_limits, n_parameters, argc, argv, &n_run_configurations, &n_total_configurations);
    
    
    if(perform_self_test()){
        cl_device_id device = get_selected_device();
        
        print_comment(device, argv);
        
        double time = convolve_ocl(padded_input, padded_output, filter, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING, device, global_config);
        copy_from_padded(output, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
        if(compare(output, output_gold, (IMAGE_WIDTH)*(IMAGE_HEIGHT)))
            printf("Self test successfull, time: %f\n", time);
        if(get_output_file() != NULL){
            printf("Writing output to %s\n", get_output_file());
            write_image_raw_float(get_output_file(), output, IMAGE_WIDTH, IMAGE_HEIGHT);
        }
    }
    else{
        
        run_on_configurations(configurations,
                              n_run_configurations,
                              n_total_configurations,
                              padded_input,
                              filter,
                              padded_output_gold,
                              argv);
    }
    
    //cl_device_id device = get_device(CL_DEVICE_TYPE_CPU);
    
    //convolve_ocl(padded_input, padded_output, filter, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING, device, global_config);
    
    //write_image_raw_float("pic.bin", padded_output, IMAGE_WIDTH+2*PADDING, IMAGE_HEIGHT+2*PADDING);
    
    //copy_from_padded(output, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    
    //print2d(output, IMAGE_WIDTH, IMAGE_HEIGHT);
    
    //compare(output, output_gold, (IMAGE_WIDTH)*(IMAGE_HEIGHT));
}


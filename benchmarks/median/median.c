// Copyright (c) 2016, Thomas L. Falch
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
int ALGORITHM =         6;
int LOCAL_FOR_SORT =    7;

int global_config[] = {4,4,1,1,0,0,0,0};
int param_limits[] =  {12,12,12,12,2,2,2,2}; //Or, rather, the limit + 1
int n_parameters = 8;

int size_map[] = {1,2,4,6,8,12,16,24,32,48,64,128};

//Problem parameters
//const int IMAGE_WIDTH = 2048;
//const int IMAGE_HEIGHT = 2048;
//const int PADDING = 4;
//const int FILTER_WIDTH = 5;
//const int FILTER_HEIGHT = 5;

//#define IMAGE_WIDTH
//#define IMAGE_HEIGHT
//#define FILTER_WIDTH
//#define FILTER_HEIGHT
//#define PADDING

unsigned char* padded_input_g;
unsigned char* filter_g;
cl_device_id device_g;


unsigned char* copy_to_padded(unsigned char* input,  int width, int height, int padding){
	unsigned char* padded_input = (unsigned char*)calloc(sizeof(unsigned char), (width+2*padding) * (height+2*padding));
	for(int i = 0; i < height; i++){
    	for(int j = 0; j < width; j++){
    		padded_input[padding*(width+2*padding) + i*(width+2*padding) + padding+ j] = input[i*width + j];
    	}
	}
	return padded_input;
}

void copy_from_padded(unsigned char* output, unsigned char* padded_output, int width, int height, int padding){
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      output[i*width+j] = padded_output[padding*(width+2*padding) + i*(width+2*padding) + padding+ j];
    }
  }
}

unsigned char* create_input(int width, int height){
	unsigned char* input = (unsigned char*)malloc(sizeof(unsigned char)*width*height);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
		  input[i*width + j] =( ((i/50)%2)*50 + ((j/70)%2)*20 + (j+i)) % 255;
          if(rand() % 20 == 0){
            input[i*width + j]  += rand() % 255;
          }
		}
	}
	return input;
}

int index(int x, int y, int width, int padding){
    return padding*(width+2*padding) + y*(width+2*padding) + x + padding;
}

void median_cpu(unsigned char* input, unsigned char* output, int width, int height, int padding){

  for(int w = 0; w < width; w++){
      for(int h = 0; h < height; h++){

          int fw = floor(FILTER_WIDTH/2);
          int fh = floor(FILTER_HEIGHT/2);

          int to_sort[FILTER_WIDTH*FILTER_HEIGHT];
          int to_sort_index = 0;

          for(int i = -fw; i <= fw; i++){
              for(int j = -fh; j <= fh; j++){


                  to_sort[to_sort_index] = input[index(w+i,h+j,width,padding)];
                  to_sort_index++;
              }
          }

          for(int i = 0; i < FILTER_WIDTH*FILTER_HEIGHT; i++){
              for(int j = i; j < FILTER_WIDTH*FILTER_HEIGHT; j++){
                  if(to_sort[j] < to_sort[i]){
                      int swap = to_sort[j];
                      to_sort[j] = to_sort[i];
                      to_sort[i] = swap;
                  }
              }
          }
          output[index(w,h,width,PADDING)] = to_sort[FILTER_WIDTH*FILTER_HEIGHT/2];
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

double median_ocl(unsigned char* input, unsigned char* output, int width, int height, int padding, cl_device_id device, int* config){
    
    int lwsx = size_map[config[LOCAL_SIZE_X]];
    int lwsy = size_map[config[LOCAL_SIZE_Y]];
    int eptx = size_map[config[ELEMENTS_PER_THREAD_X]];
    int epty = size_map[config[ELEMENTS_PER_THREAD_Y]];
    const size_t local_work_size[2] = {lwsx,lwsy};
    int gwsx = (IMAGE_WIDTH/eptx);
    int gwsy = (IMAGE_HEIGHT/epty);
    if(gwsx % lwsx != 0){
        gwsx = (gwsx/lwsx + 1)*lwsx;
    }
    if(gwsy % lwsy != 0){
        gwsy = (gwsy/lwsy+ 1)*lwsy;
    }
    const size_t global_work_size[2] = {gwsx,gwsy};
    //printf("%d, %d, %d, %d, %d, %d\n", lwsx, lwsy, eptx, epty, gwsx, gwsy);
    
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
    
    char* kernelName = "median.cl";
    char options_buffer [300];
    sprintf(options_buffer, "-D ELEMENTS_PER_THREAD_X=%d -D ELEMENTS_PER_THREAD_Y=%d"
    " -D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D FILTER_WIDTH=%d -D FILTER_HEIGHT=%d"
    " -D USE_TEXTURE=%d -D USE_LOCAL=%d -D PADDING=%d "
    " -D ALGORITHM=%d -D LOCAL_FOR_SORT=%d",
            size_map[config[ELEMENTS_PER_THREAD_X]],
            size_map[config[ELEMENTS_PER_THREAD_Y]],
            size_map[config[LOCAL_SIZE_X]],
            size_map[config[LOCAL_SIZE_Y]],
            FILTER_WIDTH,
            FILTER_HEIGHT,
            config[USE_TEXTURE],
            config[USE_LOCAL],
            PADDING,
            config[ALGORITHM],
            config[LOCAL_FOR_SORT]
    );
    
    kernel = buildKernel(kernelName, "median", options_buffer, context, device, &error);
    if(error != CL_SUCCESS){
        clReleaseKernel(kernel);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -3.0;
    }
    
    
    int align = 4096/8;
    int pitch = ((width+2*PADDING)*sizeof(unsigned char)/align);
    pitch += ((width+2*PADDING)*sizeof(unsigned char)%align) == 0 ? 0 : 1;
    pitch *= align;
    int pitched_size = pitch*(height+2*PADDING);
    size_t buffer_origin[3] = {0,0,0};
    size_t host_origin[3] = {0,0,0};
    size_t region[3] = {(width+2*PADDING)*sizeof(unsigned char), (height+2*PADDING), 1};
    
    cl_mem input_device;
    if( config[USE_TEXTURE]){
        cl_image_format image_format;
        image_format.image_channel_order = CL_R;
        image_format.image_channel_data_type = CL_UNSIGNED_INT8;
        input_device = clCreateImage2D(context,
                                       CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                       &image_format,
                                       width+2*PADDING,
                                       height+2*PADDING,
                                       (width+2*PADDING)*sizeof(unsigned char),
                                       input,
                                       &error);
    }else{
        input_device = clCreateBuffer(context,
                                      CL_MEM_READ_WRITE,
                                      pitched_size*sizeof(unsigned char),
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
    cl_mem output_device = clCreateBuffer(context, CL_MEM_READ_WRITE, pitched_size*sizeof(unsigned char), NULL, &error);
    clError("Error allocating memory",error);
    
    
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_device);
    clError("Error setting kernel argument 0",error);
    
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_device);
    clError("Error setting kernel argument 1",error);
    
    error = clSetKernelArg(kernel, 2, sizeof(cl_int), &height);
    clError("Error setting kernel argument 3",error);
    
    error = clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
    clError("Error setting kernel argument 4",error);
    
    error = clSetKernelArg(kernel, 4, sizeof(cl_int), &padding);
    clError("Error setting kernel argument 5",error);
    
    int pitch_in_floats = pitch/sizeof(unsigned char);
    error = clSetKernelArg(kernel, 5, sizeof(cl_int), &pitch_in_floats);
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

int compare(unsigned char * a, unsigned char * b, int length){
    
    if(a == NULL || b == NULL){
        return 1;
    }
    
    int n_errors = 0;
    for(int i = 0; i < length; i++){
        int diff = fabs(a[i] - b[i]);
        float rel_diff = diff/fmax(fabs(a[i]), fabs(b[i]));
        if(rel_diff > 1e-5){
            fprintf(stderr,"Error at: %d: %d %.12f, %u, %u\n", i, diff, rel_diff, a[i], b[i]);
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
                           unsigned char* padded_input,
                           unsigned char* padded_correct_output,
                           char** argv){
    
    unsigned char* padded_output = (unsigned char*)malloc(sizeof(unsigned char)*(IMAGE_WIDTH+(2*PADDING))*(IMAGE_HEIGHT+(2*PADDING)));
    unsigned char* output = (unsigned char*)calloc(sizeof(unsigned char),IMAGE_WIDTH*IMAGE_HEIGHT);
    unsigned char* correct_output = NULL;
    if(padded_correct_output){
        correct_output = (unsigned char*)calloc(sizeof(unsigned char),IMAGE_WIDTH*IMAGE_HEIGHT);
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
        


        double time = median_ocl(padded_input, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING, device, temp_config);
        copy_from_padded(output, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);


        if(time > 0 && correct_output){
            if(!compare(output, correct_output,IMAGE_HEIGHT*IMAGE_WIDTH)){
                time = -2.0;
            }
        }

        if(get_print_problem_sizes()){
            printf("%d ", IMAGE_WIDTH);
            printf("%d ", IMAGE_HEIGHT);
            printf("%d ", FILTER_WIDTH);
            printf("%d ", FILTER_HEIGHT);
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

    if(FILTER_WIDTH*FILTER_HEIGHT > 256){
        fprintf(stderr, "ERRORS!! histogram size\n");
        exit(-1);
    }
    
    parse_args(argc, argv);

    unsigned char* input = create_input(IMAGE_WIDTH, IMAGE_HEIGHT);
    unsigned char* padded_input = copy_to_padded(input, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    
    unsigned char* output = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_WIDTH*IMAGE_HEIGHT);
    unsigned char* padded_output = (unsigned char*)malloc(sizeof(unsigned char)*(IMAGE_WIDTH+2*PADDING)*(IMAGE_HEIGHT+2*PADDING));
    
    unsigned char* output_gold = NULL;
    unsigned char* padded_output_gold= NULL;
    
    
    if(get_correct_file() != NULL){
        output_gold = load_raw_buffer(get_correct_file(), IMAGE_WIDTH*IMAGE_HEIGHT);
        padded_output_gold = copy_to_padded(output_gold, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
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
        
        double time = median_ocl(padded_input, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING, device, global_config);
        copy_from_padded(output, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
        if(compare(output, output_gold, (IMAGE_WIDTH)*(IMAGE_HEIGHT)))
            printf("Self test successfull, time: %f\n", time);
        if(get_output_file() != NULL){
            printf("Writing output to %s\n", get_output_file());
            write_raw_buffer(get_output_file(), output, IMAGE_WIDTH * IMAGE_HEIGHT);
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
    

    /*
    cl_device_id device = get_device(CL_DEVICE_TYPE_GPU);
    double t = median_ocl(padded_input, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING,device, global_config);
    copy_from_padded(output, padded_output, IMAGE_WIDTH, IMAGE_HEIGHT, PADDING);
    fprintf(stderr, "%f\n", t);
    write_ppm_uchar(output, IMAGE_WIDTH, IMAGE_HEIGHT);
    */
    
}


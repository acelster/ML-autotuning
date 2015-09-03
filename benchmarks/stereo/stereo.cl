// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#ifndef __OPENCL_VERSION__
#define __constant
#define __kernel
#define __global
#define __read_only
#endif

#if USE_TEXTURE_LEFT
    #define LEFT_IMAGE_TYPE __read_only image2d_t
#else
    #define LEFT_IMAGE_TYPE __global int*
#endif


#if USE_TEXTURE_RIGHT
    #define RIGHT_IMAGE_TYPE __read_only image2d_t
#else
    #define RIGHT_IMAGE_TYPE __global int*
#endif

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


int read_left_image(LEFT_IMAGE_TYPE li, int x, int y){
    #if USE_TEXTURE_LEFT
    return read_imagei(li, imageSampler, (int2)(x,y)).x;
    #else
    return li[y*IMAGE_WIDTH+x];
    #endif
}


int read_right_image(RIGHT_IMAGE_TYPE ri, int x, int y){
    #if USE_TEXTURE_RIGHT
    return read_imagei(ri, imageSampler, (int2)(x,y)).x;
    #else
    return ri[y*IMAGE_WIDTH+x];
    #endif
}


__kernel void stereo(LEFT_IMAGE_TYPE left_image, RIGHT_IMAGE_TYPE right_image, __global int* disparity){
    
    int base_x = (get_global_id(0)*ELEMENTS_PER_THREAD_X);
    int base_y = (get_global_id(1)*ELEMENTS_PER_THREAD_Y);
    
#if USE_LOCAL_LEFT
    const int local_left_width = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS;
    const int local_left_height = LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS;
    
    __local int local_left[LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS][LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS];
    
    int elements_to_load = local_left_width * local_left_height;
    int lid = get_local_id(1) * LOCAL_SIZE_X + get_local_id(0);
    int threads_in_block = LOCAL_SIZE_X*LOCAL_SIZE_Y;
    
    
    for(int i = lid; i < elements_to_load; i+= threads_in_block){
        
        int local_row = i/(local_left_width);
        int local_col = i%(local_left_width);
        int row = local_row - RADIUS + get_group_id(1)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y);
        int col = local_col - RADIUS + get_group_id(0)*(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X);
        
        if(col < 0) col = 0;
        if(col >= IMAGE_WIDTH) col = IMAGE_WIDTH-1;
        if(row < 0) row = 0;
        if(row >= IMAGE_HEIGHT) row = IMAGE_HEIGHT-1;
        
        local_left[local_col][local_row] = read_left_image(left_image, col, row);
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
#endif
    
    #if USE_LOCAL_RIGHT
    const int local_right_width = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS;
    const int local_right_height = LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS;
    __local int local_right[LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS][LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS];
    #endif
    
    for(int x = base_x; x < base_x + ELEMENTS_PER_THREAD_X; x++){
        for(int y = base_y; y < base_y + ELEMENTS_PER_THREAD_Y; y++){
    
            int min = 999999;
            int min_d = 0;

            
            
            #pragma unroll UNROLL_DISPARITY_LOOP_FACTOR
            for(int d = MIN_DISPARITY; d <= MAX_DISPARITY; d++){

                
                #if USE_LOCAL_RIGHT
                int elements_to_load = local_right_width * local_right_height;
                int lid = get_local_id(1) * LOCAL_SIZE_X + get_local_id(0);
                int threads_in_block = LOCAL_SIZE_X*LOCAL_SIZE_Y;
                
                
                for(int i = lid; i < elements_to_load; i+= threads_in_block){
                    
                    int local_row = i/(local_right_width);
                    int local_col = i%(local_right_width);
                    int row = local_row - RADIUS + get_group_id(1)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y);
                    int col = local_col - RADIUS + get_group_id(0)*(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X) + d;
                    
                    if(col < 0) col = 0;
                    if(col >= IMAGE_WIDTH) col = IMAGE_WIDTH-1;
                    if(row < 0) row = 0;
                    if(row >= IMAGE_HEIGHT) row = IMAGE_HEIGHT-1;
                      
                    local_right[local_col][local_row] = read_right_image(right_image, col, row);
                }
                
                barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
                #endif
                
                
                
                int sum = 0;
#pragma unroll UNROLL_RADIUS_X_FACTOR
                for(int i = -RADIUS; i <= RADIUS; i++){
#pragma unroll UNROLL_RADIUS_Y_FACTOR
                    for(int j = -RADIUS; j <= RADIUS; j++){
                        
                        int xxd = x + i + d;
                        int xx = x + i;
                        int yy = y + j;
                  
                
                        #if (USE_LOCAL_LEFT || USE_LOCAL_RIGHT)
                        int lxx = xx - get_group_id(0)*(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X) + RADIUS;
                        int lyy = yy - get_group_id(1)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y) + RADIUS;
                        int lxxd = xxd - get_group_id(0)*(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X) + RADIUS;
                        
                        if( lxx >= LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS) lxx = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS -1;
                        if( lxx < 0) lxx = 0;
                        
                        if( lxxd >= LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS) lxxd = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*RADIUS -1;
                        if( lxxd < 0) lxxd = 0;
                        
                        if( lyy >= LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS) lyy = LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*RADIUS-1;
                        if( lyy < 0) lyy = 0;
                        #endif
                        
                        if( xx >= IMAGE_WIDTH) xx = IMAGE_WIDTH-1;
                        if( xx < 0) xx = 0;
                        
                        if( xxd >= IMAGE_WIDTH) xxd = IMAGE_WIDTH-1;
                        if( xxd < 0) xxd = 0;
                        
                        if( yy >= IMAGE_HEIGHT) yy = IMAGE_HEIGHT-1;
                        if( yy < 0) yy = 0;
                        
                        #if USE_LOCAL_LEFT
                        int lp = local_left[lxx][lyy];
                        #else
                        int lp = read_left_image(left_image, xx, yy);
                        #endif
                        
                        #if USE_LOCAL_RIGHT
                        int rp = local_right[lxx][lyy];
                        #else
                        int rp = read_right_image(right_image, xxd, yy);
                        #endif
                        
                        unsigned char* left_pixel = (unsigned char*)&lp;
                        unsigned char* right_pixel = (unsigned char*)&rp;
                        int absdiff = 0;
                        for (int k=0; k<4; k++){
                            absdiff += abs((int)(left_pixel[k] - right_pixel[k]));
                        }
                        sum += absdiff;
                    }
                }
                
#if USE_LOCAL_RIGHT
                barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
#endif
                
                
                
                if(sum < min){
                    min = sum;
                    min_d = d;
                }
                
            }

            disparity[y*IMAGE_WIDTH + x] = ((min_d+MAX_DISPARITY)*5);
        }
    }
}

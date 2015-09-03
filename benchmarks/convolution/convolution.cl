// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


inline int index(int x, int y, int pitch, int padding){ return padding*(pitch) + y*(pitch) + x + padding;}
inline int get_global_index_for_local(int x, int y, int lidx, int lidy, int lsx, int lsy, int width, int padding){ return index(lidx*lsx-padding+x, lidy*lsy-padding+y, width, padding);}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


#if USE_TEXTURE
inline float get(int inx, int iny, int width, int height, int pitch, int padding, __read_only image2d_t input){
#else
inline float get(int inx, int iny, int width, int height, int pitch, int padding, __global float* input){
#endif
#if FAKE_PADDING
    if(inx < 0 || iny < 0 || inx >= width || iny >= height){
        return 0.0;
    }
#endif
#if USE_TEXTURE
    return (read_imagef(input, sampler, (int2)((inx)+padding,(iny)+padding)).x);
#else
    return input[index((inx), (iny), pitch, padding)];
#endif
}
        

#if USE_TEXTURE
__kernel void convolve(__read_only image2d_t input,
                       __global float* output,
                       __constant float* filter,
                       int height,
                       int width,
                       int padding,
                       int pitch
                      ) {
#else
    __kernel void convolve(__global float* input,
                           __global float* output,
                           __constant float* filter,
                           int height,
                           int width,
                           int padding,
                           int pitch
                          ) {
#endif
    
    int x = get_global_id(0); int y = get_global_id(1);
    
#if USE_LOCAL
    __local float local_buffer[(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*PADDING)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*PADDING)];
    
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int lid = ly * LOCAL_SIZE_X + lx;
    
    int elements_to_load = (LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*padding)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*padding);
    int threads_in_block = LOCAL_SIZE_X*LOCAL_SIZE_Y;
    
    
    
    for(int i = 0; i < (elements_to_load + threads_in_block); i += threads_in_block){
        if(i + lid < elements_to_load){
            int row = (((i+lid)/(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*padding)) -padding) + get_group_id(1)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y);
            
            int col = (((i+lid)%(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*padding)) -padding ) + get_group_id(0)*(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X);
            
            //local_buffer[i + lid] = GET(col,row);
            local_buffer[i + lid] = get(col,row, width, height, pitch, PADDING, input);
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
    
    
#endif
    
    
    for(int sx = 0; sx < ELEMENTS_PER_THREAD_X; sx++){
        for(int sy = 0; sy < ELEMENTS_PER_THREAD_Y; sy++){
            
#if USE_LOCAL
            #if INTERLEAVED
            //Interleaved
            int llx = lx + get_local_size(0) * sx;
            int lly = ly + get_local_size(1) * sy;
            
            //Interleaved local
            int tx =  get_group_id(0)*get_local_size(0)*ELEMENTS_PER_THREAD_X + lx + get_local_size(0)*sx;
            int ty =  get_group_id(1)*get_local_size(1)*ELEMENTS_PER_THREAD_Y + ly + get_local_size(1)*sy;
            #else
            
            //Contigous
            int llx = lx * ELEMENTS_PER_THREAD_X + sx;
            int lly = ly * ELEMENTS_PER_THREAD_Y + sy;
            
            //Contigous
            int tx = x * ELEMENTS_PER_THREAD_X + sx;
            int ty = y * ELEMENTS_PER_THREAD_Y + sy;
            #endif
            
            
#else
            #if INTERLEAVED
            //Interleaved
            int tx = x + get_global_size(0) * sx;
            int ty = y + get_global_size(1) * sy;
            
            #else
            //Contigous
            int tx = x * ELEMENTS_PER_THREAD_X + sx;
            int ty = y * ELEMENTS_PER_THREAD_Y + sy;
            #endif
            
#endif
            
            float sum = 0.0;
            
            int k = 0;
            int fw = (FILTER_WIDTH/2);
            int fh = (FILTER_HEIGHT/2);
#if UNROLL
            #pragma unroll
#endif
            for(int i = -fw; i <= fw; i++){
#if UNROLL
                #pragma unroll
#endif
                for(int j = -fh; j <= fh; j++){
                    //sum +=  input[index(tx+i, ty+j, width, padding)] * filter[k++];
#if USE_LOCAL
                    int w = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X;
                    int idx = PADDING * (w+2*PADDING) + (lly+j)*(w +2*PADDING) + (llx+i) + PADDING;
                    sum += local_buffer[idx] * filter[k++];
                    
#else
                    //sum += GET(tx+i, ty+j)  * filter[k++];
                    sum += (get(tx+i, ty+j, width, height, pitch, padding, input)  * filter[k++]);
#endif
                   
                }
            }
            output[index(tx,ty,pitch,padding)] = sum;
        }
    }
}

// Copyright (c) 2016, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology

inline int index(int x, int y, int pitch, int padding){ return padding*(pitch) + y*(pitch) + x + padding;}
inline int get_global_index_for_local(int x, int y, int lidx, int lidy, int lsx, int lsy, int width, int padding){ return index(lidx*lsx-padding+x, lidy*lsy-padding+y, width, padding);}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


#if USE_TEXTURE
inline unsigned char get(int inx, int iny, int width, int height, int pitch, int padding, __read_only image2d_t input){
#else
inline unsigned char get(int inx, int iny, int width, int height, int pitch, int padding, __global unsigned char* input){
#endif

    if(inx >= width || iny >= height){
        return 0;
    }

#if USE_TEXTURE
    return (read_imagei(input, sampler, (int2)((inx)+padding,(iny)+padding)).x);
#else
    return input[index((inx), (iny), pitch, padding)];
#endif
}
        

#if USE_TEXTURE
__kernel void median(__read_only image2d_t input,
                       __global unsigned char* output,
                       int height,
                       int width,
                       int padding,
                       int pitch
                      ) {
#else
    __kernel void median(__global unsigned char* input,
                           __global unsigned char* output,
                           int height,
                           int width,
                           int padding,
                           int pitch
                          ) {
#endif


    
    int x = get_global_id(0); int y = get_global_id(1);

    
#if USE_LOCAL
    __local unsigned char local_buffer[(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*PADDING)*(LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*PADDING)];
    
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

    if(x*ELEMENTS_PER_THREAD_X >= width || y*ELEMENTS_PER_THREAD_Y >= height){
        return;
    }
    
    
    for(int sx = 0; sx < ELEMENTS_PER_THREAD_X; sx++){
        for(int sy = 0; sy < ELEMENTS_PER_THREAD_Y; sy++){
            
#if USE_LOCAL
            
            //Contigous
            int llx = lx * ELEMENTS_PER_THREAD_X + sx;
            int lly = ly * ELEMENTS_PER_THREAD_Y + sy;
            
            //Contigous
            int tx = x * ELEMENTS_PER_THREAD_X + sx;
            int ty = y * ELEMENTS_PER_THREAD_Y + sy;
            
            
#else
            //Contigous
            int tx = x * ELEMENTS_PER_THREAD_X + sx;
            int ty = y * ELEMENTS_PER_THREAD_Y + sy;
            
#endif
            
            
#if ALGORITHM == 0
            int fw = (FILTER_WIDTH/2);
            int fh = (FILTER_HEIGHT/2);


            int to_sort_offset = 0;
#if LOCAL_FOR_SORT
            __local unsigned char to_sort[LOCAL_SIZE_X*LOCAL_SIZE_Y*FILTER_WIDTH*FILTER_HEIGHT];
            int nlx = get_local_id(0);
            int nly = get_local_id(1);
            int nlid = nly * LOCAL_SIZE_X + nlx;
            to_sort_offset = nlid * FILTER_WIDTH*FILTER_HEIGHT;
#else
            unsigned char to_sort[FILTER_WIDTH*FILTER_HEIGHT];
#endif
            int to_sort_index = 0;


            for(int iii = 0; iii < FILTER_WIDTH*FILTER_HEIGHT; iii++){
                    int i = iii % FILTER_WIDTH;
                    int j = iii / FILTER_WIDTH;
#if USE_LOCAL
                    int w = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X;
                    int idx = PADDING * (w+2*PADDING) + (lly+(j-fh))*(w +2*PADDING) + (llx+(i-fw)) + PADDING;
                    to_sort[to_sort_offset + to_sort_index] = local_buffer[idx];
                    
#else
                    to_sort[to_sort_offset +to_sort_index] = (get(tx+(i-fw), ty+(j-fh), width, height, pitch, padding, input));
#endif
                    to_sort_index++;
            }

            for(int i = 0; i < FILTER_WIDTH*FILTER_HEIGHT/2 + 1; i++){
                for(int j = i; j < FILTER_WIDTH*FILTER_HEIGHT; j++){
                    if(to_sort[to_sort_offset +j] < to_sort[to_sort_offset +i]){
                        int swap = to_sort[to_sort_offset +j];
                        to_sort[to_sort_offset +j] = to_sort[to_sort_offset +i];
                        to_sort[to_sort_offset +i] = swap;
                    }
                }
            }

            output[index(tx,ty,pitch,padding)] = to_sort[to_sort_offset + FILTER_WIDTH*FILTER_HEIGHT/2];

#else
            int fw = (FILTER_WIDTH/2);
            int fh = (FILTER_HEIGHT/2);

            int histo_offset = 0;
#if LOCAL_FOR_SORT
            int nlx = get_local_id(0);
            int nly = get_local_id(1);
            int nlid = nly * LOCAL_SIZE_X + nlx;
            histo_offset = nlid * 256;
            __local unsigned char histo[256*LOCAL_SIZE_X*LOCAL_SIZE_Y];

#else
            unsigned char histo[256];

#endif

    
            if(sy == 0){
                for(int i = 0; i < 256; i++){
                    histo[histo_offset + i] = 0;
                }

                for(int i = -fw; i <= fw; i++){
                    for(int j = -fh; j <= fh; j++){
#if USE_LOCAL
                        int w = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X;
                        int idx = PADDING * (w+2*PADDING) + (lly+j)*(w +2*PADDING) + (llx+i) + PADDING;
                        unsigned char v = local_buffer[idx];
                        histo[histo_offset + v] += 1;

#else
                        unsigned char v = (get(tx+i, ty+j, width, height, pitch, padding, input));
                        histo[histo_offset + v] += 1;
#endif
                    }
                }
            }
            else{
                for(int i = -fw; i <= fw; i++){
                    int j = -fh - 1;
#if USE_LOCAL
                    int w = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X;
                    int idx = PADDING * (w+2*PADDING) + (lly+j)*(w +2*PADDING) + (llx+i) + PADDING;
                    unsigned char v = local_buffer[idx];
                    histo[histo_offset + v] -= 1;

#else
                    unsigned char v = (get(tx+i, ty+j, width, height, pitch, padding, input));
                    histo[histo_offset + v] -= 1;
#endif
                }
                for(int i = -fw; i <= fw; i++){
                    int j = fh;
#if USE_LOCAL
                    int w = LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X;
                    int idx = PADDING * (w+2*PADDING) + (lly+j)*(w +2*PADDING) + (llx+i) + PADDING;
                    unsigned char v = local_buffer[idx];
                    histo[histo_offset + v] += 1;

#else
                    unsigned char v = (get(tx+i, ty+j, width, height, pitch, padding, input));
                    histo[histo_offset + v] += 1;
#endif
                }
            }

            int sum = 0;
            unsigned char median_index = 0;
            sum += histo[histo_offset + median_index];
            while(sum < (FILTER_WIDTH*FILTER_HEIGHT/2 + 1)){
                median_index += 1;
                sum += histo[histo_offset + median_index];
            }

            output[index(tx,ty,pitch,padding)] = median_index;
#endif
        }
    }
}

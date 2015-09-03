// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#if USE_TEXTURE
    #define INPUT_TYPE __read_only image3d_t
#else
    #define INPUT_TYPE __global unsigned char*
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

int index(int x, int y, int z){
    int h = IMAGE_HEIGHT+ 2*PADDING;
    int w = IMAGE_WIDTH+ 2*PADDING;
    int z_base = w*h*PADDING;
    int y_base = w*PADDING;
    int x_base = PADDING;
    
    return z_base + z*h*w + y_base +  y*w + x_base + x;
}

int local_index(int x, int y, int z, int dim_x, int dim_y){
    int h = dim_y;
    int w = dim_x;
    int z_base = w*h*PADDING;
    int y_base = w*PADDING;
    int x_base = PADDING;
    
    return z_base + z*h*w + y_base +  y*w + x_base + x;
}
    


unsigned char read_input(int x, int y, int z, INPUT_TYPE input){
#if USE_TEXTURE
    return read_imagei(input, sampler, (int4)(x+PADDING,y+PADDING,z+PADDING,0)).x;
#else
    return input[index(x,y,z)];
#endif
}
        


__kernel void bilateral(INPUT_TYPE input,
                        #if PRECOMPUTE
                        __constant float* color_filter,
                        #endif
                        #if PRECOMPUTE_DIST
                        __constant float* dist_filter,
                        #endif
                        __global unsigned char* output
) {
    
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    
#if USE_LOCAL
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int lz = get_local_id(2);
    
    int lid = lz*LOCAL_SIZE_X*LOCAL_SIZE_Y +
              ly*LOCAL_SIZE_X +
              lx;
    
    int threads_in_block = LOCAL_SIZE_X*LOCAL_SIZE_Y*LOCAL_SIZE_Z;
    
    
    __local unsigned char local_buffer[(LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*PADDING)*
                                       (LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*PADDING)*
                                       (LOCAL_SIZE_Z*ELEMENTS_PER_THREAD_Z+2*PADDING)];

    int dim_x = (LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X+2*PADDING);
    int dim_y = (LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y+2*PADDING);
    int dim_z = (LOCAL_SIZE_Z*ELEMENTS_PER_THREAD_Z+2*PADDING);
    
    int dim_x_np = (LOCAL_SIZE_X*ELEMENTS_PER_THREAD_X);
    int dim_y_np = (LOCAL_SIZE_Y*ELEMENTS_PER_THREAD_Y);
    int dim_z_np = (LOCAL_SIZE_Z*ELEMENTS_PER_THREAD_Z);
    
    int elements_to_load = dim_x*dim_y*dim_z;
                           
    for(int i = 0; i < elements_to_load; i += threads_in_block){
        
        if(i + lid < elements_to_load){
            
            int slice = (lid+i) / (dim_x*dim_y) + get_group_id(2) * dim_z_np - PADDING;
            int row = ((lid+i) % (dim_x*dim_y)) / dim_x + get_group_id(1) * dim_y_np - PADDING;
            int col = ((lid+i) % (dim_x*dim_y)) % dim_x + get_group_id(0) * dim_x_np - PADDING;
            
            local_buffer[lid + i] = read_input(col, row, slice, input);
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
#endif
    
    #if OUTER_LOOP 
    for(int sx = 0; sx < ELEMENTS_PER_THREAD_X; sx++){
        for(int sy = 0; sy < ELEMENTS_PER_THREAD_Y; sy++){
            for(int sz = 0; sz < ELEMENTS_PER_THREAD_Z; sz++){
    #else
    for(int sz = 0; sz < ELEMENTS_PER_THREAD_Z; sz++){
        for(int sy = 0; sy < ELEMENTS_PER_THREAD_Y; sy++){
            for(int sx = 0; sx < ELEMENTS_PER_THREAD_X; sx++){
    #endif

                
                
                #if USE_LOCAL
                
                #if INTERLEAVED
                int llx = lx + get_local_size(0)*sx;
                int lly = ly + get_local_size(1)*sy;
                int llz = lz + get_local_size(2)*sz;
                
                int tx =  get_group_id(0)*get_local_size(0)*ELEMENTS_PER_THREAD_X + lx + get_local_size(0)*sx;
                int ty =  get_group_id(1)*get_local_size(1)*ELEMENTS_PER_THREAD_Y + ly + get_local_size(1)*sy;
                int tz =  get_group_id(2)*get_local_size(2)*ELEMENTS_PER_THREAD_Z + lz + get_local_size(2)*sz;
                
                #else
                int llx = lx*ELEMENTS_PER_THREAD_X + sx;
                int lly = ly*ELEMENTS_PER_THREAD_Y + sy;
                int llz = lz*ELEMENTS_PER_THREAD_Z + sz;
                
                int tx = x*ELEMENTS_PER_THREAD_X + sx;
                int ty = y*ELEMENTS_PER_THREAD_Y + sy;
                int tz = z*ELEMENTS_PER_THREAD_Z + sz;
                #endif
                
                #else
                
                #if INTERLEAVED
                int tx = x + get_local_size(0)* sx;
                int ty = y + get_local_size(1)* sy;
                int tz = z + get_local_size(2)* sz;
                #else
                int tx = x*ELEMENTS_PER_THREAD_X + sx;
                int ty = y*ELEMENTS_PER_THREAD_Y + sy;
                int tz = z*ELEMENTS_PER_THREAD_Z + sz;
                #endif
                
                #endif
                
                
                int i = index(tx,ty,tz);
                
                #if USE_LOCAL
                int a = local_buffer[local_index(llx, lly, llz, dim_x, dim_y)];
                #else
                int a = read_input(tx,ty,tz,input);
                #endif
                
                float norm = 0.0;
                float sum = 0.0;
                
                #if INNER_LOOP
                for(int fz = 0; fz < FILTER_DEPTH; fz++){
                    for(int fy = 0; fy < FILTER_HEIGHT; fy++){
                        for(int fx = 0; fx < FILTER_WIDTH; fx++){
                #else
                for(int fx = 0; fx < FILTER_WIDTH; fx++){
                    for(int fy = 0; fy < FILTER_HEIGHT; fy++){
                        for(int fz = 0; fz < FILTER_DEPTH; fz++){
                #endif
                            
                            int fi = fz*FILTER_WIDTH*FILTER_HEIGHT + fy * FILTER_WIDTH + fx;
                            
                            int dx = fx - FILTER_WIDTH/2;
                            int dy = fy - FILTER_HEIGHT/2;
                            int dz = fz - FILTER_DEPTH/2;
                            
                            #if USE_LOCAL
                            int b = local_buffer[local_index(llx + dx,
                                                            lly + dy,
                                                            llz + dz,
                                                            dim_x, dim_y)];
                            #else
                            int b = read_input(tx + dx, ty + dy, tz + dz, input);
                            #endif
                            
                            int color_diff = abs(a-b);
                            
                            float cf;
                            #if PRECOMPUTE
                            cf = color_filter[color_diff];
                            #else
                            cf = exp(-(color_diff*color_diff)/(2.0f*85.0f));
                            #endif
                            
                            float df;
                            #if PRECOMPUTE_DIST
                            df = dist_filter[fi];
                            #else
                            float d = sqrt((float)(dx*dx + dy*dy + dz*dz));
                            df = exp(-(d*d)/(2.0f*4.0f));
                            #endif
                            
                            norm += df*cf;
                            sum += b*df*cf;
                        }       
                    }
                }
                //output[i] = a;
                output[i] = sum/norm;
            }
        }
    }
}
    
    
    


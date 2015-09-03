// Copyright (c) 2015, Thomas L. Falch
// For conditions of distribution and use, see the accompanying LICENSE and README files

// This file is part of the benchmarks for the AUMA machine learning based auto tuning application
// developed at the Norwegian University of Science and technology


#if USE_TEXTURE_TRANSFER
    #if USE_SHARED_TRANSFER
#define TRANSTYPE __local float4*
    #else
#define TRANSTYPE __read_only image2d_t
    #endif
#else
    #if USE_SHARED_TRANSFER
#define TRANSTYPE __local float4*
    #else
        #if USE_CONSTANT_TRANSFER
#define TRANSTYPE __constant float4*
        #else
#define TRANSTYPE __global float4*
        #endif
    #endif
#endif

#if USE_TRILINEAR
__constant sampler_t dataSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
#else
__constant sampler_t dataSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
#endif

__constant sampler_t transferSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    
    return a;
}

float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;
    
    return a;
}



// Checks if position is inside the volume (float3 and int3 versions)
int inside_float(float3 pos){
    int x = (pos.x >= 0.5 && pos.x < DATA_DIM-0.5);
    int y = (pos.y >= 0.5 && pos.y < DATA_DIM-0.5);
    int z = (pos.z >= 0.5 && pos.z < DATA_DIM-0.5);
    
    return x && y && z;
}

int inside_int(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);
    
    return x && y && z;
}

       
// Indexing function (note the argument order)
inline int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Trilinear interpolation
#if USE_TEXTURE_DATA
float value_at(float3 pos, __read_only image3d_t data){
#else
float value_at(float3 pos, __global float* data){
#endif
    if(!inside_float(pos)){
        return 0;
    }
    
        
#if USE_TEXTURE_DATA
    return read_imagef(data, dataSampler, (float4)(pos.x,pos.y,pos.z,0)).x;
#else
    
    int x = floor(pos.x-0.5);
    int y = floor(pos.y-0.5);
    int z = floor(pos.z-0.5);
    
#if USE_TRILINEAR
 
    int x_u = floor(pos.x-0.5)+1;
    int y_u = floor(pos.y-0.5)+1;
    int z_u = floor(pos.z-0.5)+1;
    
    float rx = pos.x - 0.5 - x;
    float ry = pos.y - 0.5 - y;
    float rz = pos.z - 0.5 - z;
    
    
    float c0 = (1-rx) * (1-ry) * (1-rz) * data[index(z,y,x)] +
               (rx) * (1-ry) * (1-rz) * data[index(z,y,x_u)] +
               (1-rx) * (ry) * (1-rz) * data[index(z,y_u,x)] +
               (rx) * (ry) * (1-rz) * data[index(z,y_u,x_u)] +
               (1-rx) * (1-ry) * (rz) * data[index(z_u,y,x)] +
               (rx) * (1-ry) * (rz) * data[index(z_u,y,x_u)] +
               (1-rx) * (ry) * (rz) * data[index(z_u,y_u,x)] +
               (rx) * (ry) * (rz) * data[index(z_u,y_u,x_u)];
    
    return c0;
    
#else
    return data[index(z,y,x)];
#endif
#endif
}

#if USE_TEXTURE_DATA
float4 color_at(float3 pos, __read_only image3d_t data, TRANSTYPE transfer){
#else
float4 color_at(float3 pos, __global float* data, TRANSTYPE transfer){
#endif
    
    float v = value_at(pos,data);
    float4 c = (float4)(0,0,0,0);
    if(v == 0){
        return c;
    }
    else if(v > 1.0f){
        return c;
    }
    int i = (int)(v * (float)(TRANSFER_FUNC_SIZE-1));
    
                    
    
   
#if USE_TEXTURE_TRANSFER && !USE_SHARED_TRANSFER
    return read_imagef(transfer, transferSampler, (int2)(i,1));
#else    
    return transfer[i];
#endif    
}

int rgbaToInt(float4 color){
    color.x *= color.w;
    color.y *= color.w;
    color.z *= color.w;
    
    color.x = color.x > 1.0 ? 1.0 : color.x;
    color.y = color.y > 1.0 ? 1.0 : color.y;
    color.z = color.z > 1.0 ? 1.0 : color.z;
    
    return ((uint)(color.w*255)<<24) | ((uint)(color.z*255)<<16) | ((uint)(color.y*255)<<8) | (uint)(color.x*255);
}

int intersectBox(float3 ray, float3 camera, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    float3 invR = ((float3)(1.0,1.0,1.0)) / ray;
    float3 tbot = invR * (boxmin - camera);
    float3 ttop = invR * (boxmax - camera);

    float3 tmin = min(ttop, tbot);
    float3 tmax = max(ttop, tbot);

    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

 
#if USE_TEXTURE_DATA
#if USE_TEXTURE_TRANSFER
__kernel void raycast(__read_only image3d_t data, __global int* image, __read_only image2d_t transfer){
#else
    #if USE_CONSTANT_TRANSFER
__kernel void raycast(__read_only image3d_t data, __global int* image, __constant float4* transfer){
    #else
__kernel void raycast(__read_only image3d_t data, __global int* image, __global float4* transfer){    
    #endif
#endif
#else
#if USE_TEXTURE_TRANSFER
__kernel void raycast(__global float* data, __global int* image, __read_only image2d_t transfer){
#else
    #if USE_CONSTANT_TRANSFER
__kernel void raycast(__global float* data, __global int* image, __constant float4* transfer){
    #else
__kernel void raycast(__global float* data, __global int* image, __global float4* transfer){
    #endif
#endif
#endif
    
#if USE_SHARED_TRANSFER
    __local float4 localTransfer[TRANSFER_FUNC_SIZE];
    
    int lsize = get_local_size(0) * get_local_size(1);
    int local_id = get_local_size(0)*get_local_id(1) + get_local_id(0);
    
    for(int i = local_id; i < TRANSFER_FUNC_SIZE; i+= lsize){    
#if USE_TEXTURE_TRANSFER
        localTransfer[i] = read_imagef(transfer, transferSampler, (int2)(i,1));
#else    
        localTransfer[i] = transfer[i];
#endif
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
#endif
    

    
    float3 camera = (float3)(500,500,500);
    float3 forward = (float3)(-1, -1, -1);
    float3 z_axis = (float3)(0, 0, 1);
    
    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);
    
    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);
    
    float fov = 3.14/6;
    float pixel_width = tan(fov/2.0)/(IMAGE_WIDTH/2);
    float pixel_height = pixel_width;
    float step_size = 0.5;
    float3 screen_center = camera + forward;
    
#if INTERLEAVED
    
    int base_x = get_global_id(0) - (IMAGE_WIDTH/2);
    int base_y = get_global_id(1) - (IMAGE_HEIGHT/2);
    int x_step = IMAGE_WIDTH/ELEMENTS_PER_THREAD_X;
    int y_step = IMAGE_HEIGHT/ELEMENTS_PER_THREAD_Y;
    
    for(int x = base_x; x < IMAGE_WIDTH/2; x += x_step){
        for(int y = base_y; y < IMAGE_HEIGHT/2; y += y_step){
#else
    int base_x = (get_global_id(0)*ELEMENTS_PER_THREAD_X) - (IMAGE_WIDTH/2);
    int base_y = (get_global_id(1)*ELEMENTS_PER_THREAD_Y) - (IMAGE_HEIGHT/2);
    
    for(int x = base_x; x < base_x + ELEMENTS_PER_THREAD_X; x++){
        for(int y = base_y; y < base_y + ELEMENTS_PER_THREAD_Y; y++){
#endif

            
            float3 ray = screen_center + right*x*pixel_width + up*y*pixel_height;
            ray = ray - camera;
            ray = normalize(ray);
            
            float near, far;
            intersectBox(ray, camera, ((float3)(0,0,0)), ((float3)(DATA_DIM,DATA_DIM,DATA_DIM)), &near, &far);
            float3 pos = camera + near * ray;
            
            float i = 0;
            float4 color = (float4)(0,0,0,0);
            while(color.w < 1.0 && i < (far-near)){
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                float4 c = color_at(pos, data, localTransfer);
#else
                float4 c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                

                

                
#if UNROLL_FACTOR > 1
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
#endif                
#if UNROLL_FACTOR > 2                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 3                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 4                                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 5                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 6                                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 7                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
                
                                
#endif                
#if UNROLL_FACTOR > 8                                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 9                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
                                
#endif                
#if UNROLL_FACTOR > 10                 
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 11                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
                
                                
#endif                
#if UNROLL_FACTOR > 12                                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 13                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 14                                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                
#if UNROLL_FACTOR > 15                 
                
                
                i += step_size; 
                pos = pos + ray*step_size;
#if USE_SHARED_TRANSFER
                c = color_at(pos, data, localTransfer);
#else
                c = color_at(pos, data, transfer);
#endif
                c.x *= c.w;
                c.y *= c.w;
                c.z *= c.w;
                color = color + (1.0f-color.w)*c;
                
#endif                    

            }
    
            image[(y+(IMAGE_HEIGHT/2)) * IMAGE_WIDTH + (x+(IMAGE_WIDTH/2))] = rgbaToInt(color);
        }
    }
}
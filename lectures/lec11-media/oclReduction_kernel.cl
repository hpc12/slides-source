/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/

// The following defines are set during runtime compilation, see reduction.cpp
// #define T float
// #define GROUP_SIZE 128
// #define nIsPow2 1

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
// red0
__kernel void reduce0(__global T *g_idata, __global T *g_odata, 
    unsigned int n, __local T* ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int i = get_global_id(0);

    ldata[lid] = (i < n) ? g_idata[i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int s=1; s < get_local_size(0); s *= 2) 
    {
        if ((lid % (2*s)) == 0)
            ldata[lid] += ldata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
}
// end


/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts. */
// red1
__kernel void reduce1(__global T *g_idata, __global T *g_odata, 
    unsigned int n, __local T* ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int i = get_global_id(0);

    ldata[lid] = (i < n) ? g_idata[i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int s=1; s < get_local_size(0); s *= 2)
    {
        int index = 2 * s * lid;
        if (index < get_local_size(0))
            ldata[index] += ldata[index + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
}
// end

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
// red2
__kernel void reduce2(__global T *g_idata, __global T *g_odata, 
    unsigned int n, __local T* ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int i = get_global_id(0);

    ldata[lid] = (i < n) ? g_idata[i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1)
    {
        if (lid < s)
            ldata[lid] += ldata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) g_odata[get_local_size(0)] = ldata[0];
}
// end

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory
*/
// red3
__kernel void reduce3(__global T *g_idata, __global T *g_odata, 
    unsigned int n, __local T* ldata)
{
    unsigned lid = get_local_id(0);
    unsigned i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

    ldata[lid] = (i < n) ? g_idata[i] : 0;
    if (i + get_local_size(0) < n)
        ldata[lid] += g_idata[i+get_local_size(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned s=get_local_size(0)/2; s>0; s>>=1)
    {
        if (lid < s)
            ldata[lid] += ldata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
}
// end

/*
    This version unrolls the last warp to avoid synchronization where it
    isn't needed
*/
// red4p1
__kernel void reduce4(__global T *g_idata, __global T *g_odata, 
    unsigned int n, volatile __local T* ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) 
      + get_local_id(0);

    ldata[lid] = (i < n) ? g_idata[i] : 0;
    if (i + get_local_size(0) < n)
        ldata[lid] += g_idata[i+get_local_size(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int s=get_local_size(0)/2; s>32; s>>=1)
    {
        if (lid < s)
            ldata[lid] += ldata[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
// end
// red4p2
    if (lid < 32)
    {
        if (GROUP_SIZE >=  64) { ldata[lid] += ldata[lid + 32]; }
        if (GROUP_SIZE >=  32) { ldata[lid] += ldata[lid + 16]; }
        //..
        if (GROUP_SIZE >=   2) { ldata[lid] += ldata[lid +  1]; }
    }

    if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
}
// end

/*
    This version is completely unrolled.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads.  This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time.
*/
__kernel void reduce5(__global T *g_idata, __global T *g_odata, unsigned int n, __local T* ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

    ldata[lid] = (i < n) ? g_idata[i] : 0;
    if (i + GROUP_SIZE < n)
        ldata[lid] += g_idata[i+GROUP_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (GROUP_SIZE >= 512) { if (lid < 256) { ldata[lid] += ldata[lid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (GROUP_SIZE >= 256) { if (lid < 128) { ldata[lid] += ldata[lid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (GROUP_SIZE >= 128) { if (lid <  64) { ldata[lid] += ldata[lid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }

    if (lid < 32)
    {
        if (GROUP_SIZE >=  64) { ldata[lid] += ldata[lid + 32]; }
        if (GROUP_SIZE >=  32) { ldata[lid] += ldata[lid + 16]; }
        if (GROUP_SIZE >=  16) { ldata[lid] += ldata[lid +  8]; }
        if (GROUP_SIZE >=   8) { ldata[lid] += ldata[lid +  4]; }
        if (GROUP_SIZE >=   4) { ldata[lid] += ldata[lid +  2]; }
        if (GROUP_SIZE >=   2) { ldata[lid] += ldata[lid +  1]; }
    }

    if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
*/
// red6p1
__kernel void reduce6(__global T *g_idata, __global T *g_odata, 
    unsigned int n, volatile __local T* ldata)
{
    unsigned int lid = get_local_id(0);
    unsigned int i = get_group_id(0)*(
        get_local_size(0)*2) + get_local_id(0);
    unsigned int gridSize = GROUP_SIZE*2*get_num_groups(0);
    ldata[lid] = 0;

    while (i < n)
    {
        ldata[lid] += g_idata[i];
        if (i + GROUP_SIZE < n)
            ldata[lid] += g_idata[i+GROUP_SIZE];
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
// end
//
// red6p2
    if (GROUP_SIZE >= 512) 
    { 
      if (lid < 256) { ldata[lid] += ldata[lid + 256]; } 
      barrier(CLK_LOCAL_MEM_FENCE); 
    }
    // ...
    if (GROUP_SIZE >= 128) 
    { /* ... */ }

    if (lid < 32)
    {
        if (GROUP_SIZE >=  64) { ldata[lid] += ldata[lid + 32]; }
        if (GROUP_SIZE >=  32) { ldata[lid] += ldata[lid + 16]; }
        // ...
        if (GROUP_SIZE >=   2) { ldata[lid] += ldata[lid +  1]; }
    }

    if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
}
// end

#endif // #ifndef _REDUCE_KERNEL_H_

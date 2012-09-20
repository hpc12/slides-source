__kernel void twice(__global float *a)
{ a[get_global_id(0)] *= 2; }

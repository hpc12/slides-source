// slide1
#include "cl-helper.h"

int main()
{
  // init
  cl_context ctx; cl_command_queue queue;
  create_context_on("NVIDIA", NULL, 0, &ctx, &queue, 0);

  // allocate and initialize CPU memory
  const size_t sz = 10000;
  float a[sz];
  for (size_t i = 0; i < sz; ++i) a[i] = i;
// end
// slide2
  // allocate GPU memory, transfer to GPU

  cl_int status;
  cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
      sizeof(float) * sz, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
        sz * sizeof(float), a,
        0, NULL, NULL));
// end
// slide3
  // load kernels 
  char *knl_text = read_file("twice.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "twice", NULL);
  free(knl_text);

  // run code on GPU
  SET_1_KERNEL_ARG(knl, buf_a);
  size_t gdim[] = { sz };
  size_t ldim[] = { 1 };
  CALL_CL_GUARDED(clEnqueueNDRangeKernel,
      (queue, knl,
       /*dimensions*/ 1, NULL, gdim, ldim,
       0, NULL, NULL));
// end
// slide4
  // clean up...
  CALL_CL_GUARDED(clReleaseMemObject, (buf_a));
  CALL_CL_GUARDED(clReleaseKernel, (knl));
  CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
  CALL_CL_GUARDED(clReleaseContext, (ctx));
}
// end

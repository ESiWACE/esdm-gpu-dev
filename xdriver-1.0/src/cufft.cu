#include <stdio.h>
#include <cufft.h>

cufftHandle plan;
cufftResult result;

// 1D FFT single precision ====================================================

void sPlan1dCUFFT(int n, void *stream) {
  result = cufftPlan1d(&plan, n, CUFFT_C2C, 1);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftPlan1d failed: code = %i\n",result);
     return;
  }
  result = cufftSetStream(plan, (cudaStream_t)stream);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftSetStream failed: code = %i\n",result);
     return;
  }
}

// 2D FFT single precision ====================================================

void sPlan2dCUFFT(int nx, int ny, void *stream) {
  result = cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftPlan2d failed: code = %i\n",result);
     return;
  }
  result = cufftSetStream(plan, (cudaStream_t)stream);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftSetStream failed: code = %i\n",result);
     return;
  }
}

// single precision execute & destroy =========================================

void sExecCUFFT(float *sdata) {
  result = cufftExecC2C(plan, (cufftComplex*)sdata, (cufftComplex*)sdata, CUFFT_FORWARD);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftExecC2C failed: code = %i\n",result);
     return;
  }
}

void sDestroyCUFFT() {
  result = cufftDestroy(plan);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftDestroy failed: code = %i\n",result);
     return;
  }
}

// 1D FFT double precision ====================================================

void dPlan1dCUFFT(int n, void *stream) {
  result = cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftPlan1d failed: code = %i\n",result);
     return;
  }
  result = cufftSetStream(plan, (cudaStream_t)stream);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftSetStream failed: code = %i\n",result);
     return;
  }
}

// 2D FFT double precision ====================================================

void dPlan2dCUFFT(int nx, int ny, void *stream) {
  result = cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftPlan2d failed: code = %i\n",result);
     return;
  }
  result = cufftSetStream(plan, (cudaStream_t)stream);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftSetStream failed: code = %i\n",result);
     return;
  }
}

// double precision execute & destroy =========================================

void dExecCUFFT(double *ddata) {
  result = cufftExecZ2Z(plan, (cufftDoubleComplex*)ddata, (cufftDoubleComplex*)ddata, CUFFT_FORWARD);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftExecZ2Z failed: code = %i\n",result);
     return;
  }
}

void dDestroyCUFFT() {
  result = cufftDestroy(plan);
  if (result!=CUFFT_SUCCESS) { 
     printf ("Error: cufftDestroy failed: code = %i\n",result);
     return;
  }
}

/*
// Original versions from the OLCF website
// https://www.olcf.ornl.gov/tutorials/mixing-openacc-with-gpu-libraries/
//
void sLaunchCUFFT(float *sdata, int n, void *stream) {
  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, 1);
  cufftSetStream(plan, (cudaStream_t)stream);
  cufftExecC2C(plan, (cufftComplex*)sdata, (cufftComplex*)sdata, CUFFT_FORWARD);
  cufftDestroy(plan);
}

void dLaunchCUFFT(double *ddata, int n, void *stream) {
  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
  cufftSetStream(plan, (cudaStream_t)stream);
  cufftExecZ2Z(plan, (cufftDoubleComplex*)ddata, (cufftDoubleComplex*)ddata, CUFFT_FORWARD);
  cufftDestroy(plan);
}
*/

#include <cufft.h>

cufftHandle splan;
cufftHandle dplan;

void sPlanCUFFT(int n, void *stream)
{
    cufftPlan1d(&splan, n, CUFFT_C2C, 1);
    cufftSetStream(splan, (cudaStream_t)stream);
}

void sExecCUFFT(float *sdata)
{
    cufftExecC2C(splan, (cufftComplex*)sdata, (cufftComplex*)sdata, CUFFT_FORWARD);
}

void sDestroyCUFFT()
{
    cufftDestroy(splan);
}

void dPlanCUFFT(int n, void *stream)
{
    cufftPlan1d(&dplan, n, CUFFT_Z2Z, 1);
    cufftSetStream(dplan, (cudaStream_t)stream);
}

void dExecCUFFT(double *ddata)
{
    cufftExecZ2Z(dplan, (cufftDoubleComplex*)ddata, (cufftDoubleComplex*)ddata, CUFFT_FORWARD);
}

void dDestroyCUFFT()
{
    cufftDestroy(dplan);
}

/*
void sLaunchCUFFT(float *sdata, int n, void *stream)
{
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    cufftSetStream(plan, (cudaStream_t)stream);
    cufftExecC2C(plan, (cufftComplex*)sdata, (cufftComplex*)sdata, CUFFT_FORWARD);
    cufftDestroy(plan);
}

void dLaunchCUFFT(double *ddata, int n, void *stream)
{
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
    cufftSetStream(plan, (cudaStream_t)stream);
    cufftExecZ2Z(plan, (cufftDoubleComplex*)ddata, (cufftDoubleComplex*)ddata, CUFFT_FORWARD);
    cufftDestroy(plan);
}
*/

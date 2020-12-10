#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "xd_timer.h"
#include "functions.h"
#include "compare.h"

int test_varrayMinMaxMV (float *sarray, double *darray, int &nx, int &ny, int &niter,
    int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
    double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu) {

  int i, ierr, iter, j;
  int n;
  size_t nm, nsize;
  float smissval;
  double dmissval;
  double end, start;
  std::pair<float, float> sminmax;
  std::pair<double, double> dminmax;

//printf ("test_varrayMinMaxMV: timing harness for CDO operator varrayMinMaxMV\n");
//printf ("test_varrayMinMaxMV: nx %i ny %i niter %i\n",
//    nx, ny, niter);
//printf ("test_varrayMinMaxMV: cpuonly %i gpuonly %i single %i double %i\n",
//    cpuonly, gpuonly, singledata, doubledata);

  ierr = 0;

// ------------------------------------------------------------------
// SINGLE PRECISION
// ------------------------------------------------------------------

// Fill the single precision data arrays with artificial data

  n = nx*ny;
  nsize = (size_t) n;
  for (j=0;j<ny;j++) {
    for (i=0;i<nx;i++) {
      *(sarray+j*nx+i) = i+j*j/1000.0;
    }
// printf ("%f %f %f %f %f\n",sarray[j][0],sarray[j][1],sarray[j][2],sarray[j][3],sarray[j][4]);
  }

// Add in a couple of missing values

  smissval = -999.0;
  j = ny/2; i = nx/2;
  *(sarray+j*nx+i) = smissval;
  j = ny/3; i = nx/3;
  *(sarray+j*nx+i) = smissval;

// CPU code, single precision, timing loop

  if ( singledata && !gpuonly ) {
    start = TIMER_FUNCTION;

    for (iter=0;iter<niter;iter++ ) {
      sminmax = CPUs_varrayMinMaxMV(nsize,sarray,smissval,nm);
    }

    end = TIMER_FUNCTION;
    *stimecpu = end-start;

    printf ("varrayMinMaxMV, CPU, single, DBL_IS_EQUAL    with isnan: Min %f Max %f nMiss %i\n",
      sminmax.first,sminmax.second,(int) (n-nm));
  }

// GPU code, single precision, timing loop

  if ( singledata && !cpuonly ) {
    start = TIMER_FUNCTION;

    for (iter=0;iter<niter;iter++ ) {
      sminmax = GPUs_varrayMinMaxMV(nsize,sarray,smissval,nm);
    }

    end = TIMER_FUNCTION;
    *stimegpu = end-start;

    printf ("varrayMinMaxMV, GPU, single, DBL_IS_EQUAL    with isnan: Min %f Max %f nMiss %i\n",
      sminmax.first,sminmax.second,(int) (n-nm));
  }

// ------------------------------------------------------------------
// DOUBLE PRECISION
// ------------------------------------------------------------------

// Fill the double precision data arrays with artificial data

  n = nx*ny;
  for (j=0;j<ny;j++) {
    for (i=0;i<nx;i++) {
      *(darray+j*nx+i) = i+j*j/1000.0;
    }
// printf ("%f %f %f %f %f\n",darray[j][0],darray[j][1],darray[j][2],darray[j][3],darray[j][4]);
  }

// Add in a couple of missing values

  dmissval = -999.0;
  j = ny/2; i = nx/2;
  *(darray+j*nx+i) = dmissval;
  j = ny/3; i = nx/3;
  *(darray+j*nx+i) = dmissval;

// CPU code, DOUBLE precision, timing loop

  if ( doubledata && !gpuonly ) {
    start = TIMER_FUNCTION;

    for (iter=0;iter<niter;iter++ ) {
      dminmax = CPUd_varrayMinMaxMV(nsize,darray,dmissval,nm);
    }

    end = TIMER_FUNCTION;
    *dtimecpu = end-start;

    printf ("varrayMinMaxMV, CPU, double, DBL_IS_EQUAL    with isnan: Min %f Max %f nMiss %i\n",
      dminmax.first,dminmax.second,(int) (n-nm));
  }

// GPU code, DOUBLE precision, timing loop

  if ( doubledata && !cpuonly ) {
    start = TIMER_FUNCTION;

    for (iter=0;iter<niter;iter++ ) {
      dminmax = GPUd_varrayMinMaxMV(nsize,darray,dmissval,nm);
    }

    end = TIMER_FUNCTION;
    *dtimegpu = end-start;

    printf ("varrayMinMaxMV, GPU, double, DBL_IS_EQUAL    with isnan: Min %f Max %f nMiss %i\n",
      dminmax.first,dminmax.second,(int) (n-nm));
  }

  return ierr;

}

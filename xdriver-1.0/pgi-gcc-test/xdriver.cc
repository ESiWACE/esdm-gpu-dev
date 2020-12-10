#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cfloat>          // for DBL_MAX
#include <math.h>
#include <time.h>
#include <utility>         // for std::pair
#include <unistd.h>        // for gethostname
#if defined(_OPENMP)
#include <omp.h>
#endif
#define __MAIN__

#include "compare.h"

std::pair<double, double>
CPUvarrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals);
std::pair<double, double>
GPUvarrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals);

int main (int argc, char *argv[]) {

#define NX 1000
#define NY 500
#define NITER 1000
  int i,iter,j,niter;
  size_t n,nm,nx,ny;
  double end,missval,start,timecpu,timegpu;
  double a[NY][NX];
  std::pair<double, double> minmax;

  printf ("Hello from xdriver!\n");

  niter = NITER;
  nx = NX;
  ny = NY;
  n = nx*ny;
  for (j=0;j<ny;j++) {
    for (i=0;i<nx;i++) {
      a[j][i] = i+j*j/1000.0;
    }
// printf ("%f %f %f %f %f\n",a[j][0],a[j][1],a[j][2],a[j][3],a[j][4]);
  }
  missval = -999.0;
  a[ny/2][nx/2] = missval;
  a[ny/3][nx/3] = missval;

// CPU code
// timing loop
  printf("xdriver: %i iterations for timing\n",niter);
#if defined(_OPENMP)
  start =  omp_get_wtime();
#else
  start = (double) clock() / CLOCKS_PER_SEC;
#endif

  for (iter=0;iter<niter;iter++ ) {
    minmax = CPUvarrayMinMaxMV(n,*a,missval,nm);
  }

#if defined(_OPENMP)
  end = omp_get_wtime();
#else
  end = (double) clock() / CLOCKS_PER_SEC;
#endif
  timecpu = end-start;

  printf ("Minimum        %f Maximum        %f Missing values %i\n",
    minmax.first,minmax.second,(int) (n-nm));

// GPU code
// timing loop
#if defined(_OPENMP)
  start =  omp_get_wtime();
#else
  start = (double) clock() / CLOCKS_PER_SEC;
#endif

  for (iter=0;iter<niter;iter++ ) {
    minmax = GPUvarrayMinMaxMV(n,*a,missval,nm);
  }

#if defined(_OPENMP)
  end = omp_get_wtime();
#else
  end = (double) clock() / CLOCKS_PER_SEC;
#endif
  timegpu = end-start;

  printf ("Minimum        %f Maximum        %f Missing values %i\n",
    minmax.first,minmax.second,(int) (n-nm));
  printf ("Time CPU       %f\n",timecpu);
  printf ("Time GPU       %f\n",timegpu);
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "xd_timer.h"
//#include "array.h"
#include "functions.h"
#include "compare.h"
#include "dev_interpolation.h"

int test_vert_interp_lev (float *sarray, double *darray, int &nx, int &ny, int &nlev, int &niter,
    int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
    int &do_isnan, int &do_missval,
    double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu) {

  int i, ierr, ilev, iter, j;
  int n;
  size_t gridsize;
  double end, start;
  float *svardata1, *svardata2, smissval;
  float *slev_wgt1, *slev_wgt2;
  double *dvardata1, *dvardata2, dmissval;
  double *dlev_wgt1, *dlev_wgt2;
  int *lev_idx1, *lev_idx2;

//printf ("test_vert_interp_lev: timing harness for CDO operator vert_interp_lev\n");
//printf ("test_vert_interp_lev: nx %i ny %i niter %i\n",
//    nx, ny, niter);
//printf ("test_vert_interp_lev: cpuonly %i gpuonly %i single %i double %i\n",
//    cpuonly, gpuonly, singledata, doubledata);

  ierr = 0;

  n = nx*ny;
  gridsize = (size_t) n;

  dlev_wgt1 = (double*)malloc(nlev*sizeof(double));
  dlev_wgt2 = (double*)malloc(nlev*sizeof(double));
  slev_wgt1 = (float*)dlev_wgt1;
  slev_wgt2 = (float*)dlev_wgt2;
  svardata1 = sarray;
  svardata2 = sarray;
  dvardata1 = darray;
  dvardata2 = darray;

// Setup the level arrays
  lev_idx1 = (int*)malloc(nlev*sizeof(int));
  lev_idx2 = (int*)malloc(nlev*sizeof(int));
  for (ilev=0;ilev<nlev-1;ilev++) {
    *(lev_idx1+ilev) = ilev;
    *(lev_idx2+ilev) = ilev+1;
  }
  *(lev_idx1+nlev-1) = *(lev_idx1+nlev-2);
  *(lev_idx2+nlev-1) = *(lev_idx2+nlev-2);

// ------------------------------------------------------------------
// SINGLE PRECISION
// ------------------------------------------------------------------

  if ( singledata ) {

// Fill the single precision data arrays with artificial data

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

// Set the weights

    for (ilev=0;ilev<nlev;ilev++) {
      *(slev_wgt1+ilev) = 0.5+ilev*0.05;
      *(slev_wgt2+ilev) = 0.5-ilev*0.05;
    }
/*
    for (ilev=0;ilev<nlev;ilev++) {
      printf("Levels %i %i    Weights %f %f\n",*(lev_idx1+ilev),*(lev_idx2+ilev),
        *(slev_wgt1+ilev),*(slev_wgt2+ilev));
    }
*/

// CPU code, single precision, timing loop

    if ( !gpuonly ) {

      if ( do_isnan && do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          CPUs_vert_interp_lev( gridsize, smissval, svardata1, svardata2,
                    nlev, lev_idx1, lev_idx2, slev_wgt1, slev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else if ( do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          CPUs_vert_interp_lev_noisnan( gridsize, smissval, svardata1, svardata2,
                    nlev, lev_idx1, lev_idx2, slev_wgt1, slev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          CPUs_vert_interp_lev_nomissval( gridsize, smissval, svardata1, svardata2,
                    nlev, lev_idx1, lev_idx2, slev_wgt1, slev_wgt2);
        }
        end = TIMER_FUNCTION;
      }

      *stimecpu = end-start;

      printf ("vert_interp_lev, CPU, single,                           : TBD\n");
    }

// GPU code, single precision, timing loop

    if ( !cpuonly ) {

      if ( do_isnan && do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          GPUs_vert_interp_lev( gridsize, smissval, svardata1, svardata2,
                    nlev, lev_idx1, lev_idx2, slev_wgt1, slev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else if ( do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          GPUs_vert_interp_lev_noisnan( gridsize, smissval, svardata1, svardata2,
                    nlev, lev_idx1, lev_idx2, slev_wgt1, slev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          GPUs_vert_interp_lev_nomissval( gridsize, smissval, svardata1, svardata2,
                    nlev, lev_idx1, lev_idx2, slev_wgt1, slev_wgt2);
        }
        end = TIMER_FUNCTION;
      }

      *stimegpu = end-start;

      printf ("vert_interp_lev, GPU, single,                           : TBD\n");
    }

/*
    if ( singledata && !cpuonly ) {
      size_t nLev1 = (size_t) nlev;
      size_t nLev2 = (size_t) nlev;
      auto vardata1 = Varray<float>(gridsize*nlev,1);
//  *vardata1 = svardata1;
      auto vardata2 = Varray<float>(gridsize*nlev,1);
      auto lev_idx = Varray<int>(nLev1,1);
      auto lev_wgt = Varray<float>(nLev2,1);

printf("%i %i vardata1 capacity %i\n",gridsize,nlev,vardata1.capacity());
      start = TIMER_FUNCTION;

      for (iter=0;iter<niter;iter++ ) {
        vert_interp_lev(gridsize, nLev1, smissval, vardata1, vardata2, nLev2, lev_idx, lev_wgt);
      }

      end = TIMER_FUNCTION;
      *stimegpu = end-start;

      printf ("vert_interp_lev, GPU, single, templated                 : TBD\n",
    }
*/

  }

// ------------------------------------------------------------------
// DOUBLE PRECISION
// ------------------------------------------------------------------

  if ( doubledata ) {

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

// Set the weights

    for (ilev=0;ilev<nlev;ilev++) {
      *(dlev_wgt1+ilev) = 0.5+ilev*0.05;
      *(dlev_wgt2+ilev) = 0.5-ilev*0.05;
    }

// CPU code, DOUBLE precision, timing loop

    if ( !gpuonly ) {
      if ( do_isnan && do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          CPUd_vert_interp_lev( gridsize, dmissval, dvardata1, dvardata2,
                    nlev, lev_idx1, lev_idx2, dlev_wgt1, dlev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else if ( do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          CPUd_vert_interp_lev_noisnan( gridsize, dmissval, dvardata1, dvardata2,
                    nlev, lev_idx1, lev_idx2, dlev_wgt1, dlev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          CPUd_vert_interp_lev_nomissval( gridsize, dmissval, dvardata1, dvardata2,
                    nlev, lev_idx1, lev_idx2, dlev_wgt1, dlev_wgt2);
        }
        end = TIMER_FUNCTION;
      }

      *dtimecpu = end-start;

      printf ("vert_interp_lev, CPU, double,                           : TBD\n");
    }

// GPU code, DOUBLE precision, timing loop

    if ( !cpuonly ) {
      if ( do_isnan && do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          GPUd_vert_interp_lev( gridsize, dmissval, dvardata1, dvardata2,
                    nlev, lev_idx1, lev_idx2, dlev_wgt1, dlev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else if ( do_missval ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          GPUd_vert_interp_lev_noisnan( gridsize, dmissval, dvardata1, dvardata2,
                    nlev, lev_idx1, lev_idx2, dlev_wgt1, dlev_wgt2);
        }
        end = TIMER_FUNCTION;
      }
      else {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          GPUd_vert_interp_lev_nomissval( gridsize, dmissval, dvardata1, dvardata2,
                    nlev, lev_idx1, lev_idx2, dlev_wgt1, dlev_wgt2);
        }
        end = TIMER_FUNCTION;
      }

      *dtimegpu = end-start;

      printf ("vert_interp_lev, GPU, double,                           : TBD\n");
    }

  }

  return ierr;

}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "openacc.h"
#include <fftw3.h>

#include "xd_timer.h"
#include "functions.h"

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

// Forward declaration of wrapper functions that will call CUFFT
extern void sMulti2dCUFFT(int nx, int ny, float *sarray2, int ngpu);
//extern void sLaunchCUFFT(float *d_data, int n, void *stream);
extern void sPlan2dCUFFT(int nx, int ny, void *stream);
extern void sExecCUFFT(float *sarray);
extern void sDestroyCUFFT();
//extern void dLaunchCUFFT(double *d_data, int n, void *stream);
extern void dPlan2dCUFFT(int nx, int ny, void *stream);
extern void dExecCUFFT(double *darray);
extern void dDestroyCUFFT();

int test_fft2d (float *sarray, double *darray, int &nx, int &ny, int &niter,
    int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
    int &ffttype, int &ngpu,
    double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu) {

  int i, ierr, iter, j;
  int fftdatasize;
  float *sarray1, *sarray2;
  double *darray1, *darray2;
  double freqdata;
  double end, start;

//printf ("test_fft2d: timing harness for CDO operator fft\n");
//printf ("test_fft2d: nx %i ny %i niter %i\n",
//    nx, ny, niter);
//printf ("test_fft2d: cpuonly %i gpuonly %i single %i double %i\n",
//    cpuonly, gpuonly, singledata, doubledata);

  ierr = 0;
  fftw_plan p;
  fftwf_plan pf;

// The FFT is of length nx complex values
// Therefore the data size is twice nx*ny
  fftdatasize = 2*nx*ny;

  freqdata = 7.0;
// Because some FFTs compute in-place
// data checking will only take place when one iteration is set
  if ( niter==1 ) printf ("fft, frequency of input data: %6.1f\n",freqdata);

// Set up pointers to separate input and output arrays
  sarray1 = sarray;
  sarray2 = sarray+fftdatasize;
  darray1 = darray;
  darray2 = darray+fftdatasize;

// ------------------------------------------------------------------
// SINGLE PRECISION
// ------------------------------------------------------------------

  if ( singledata ) {

// Fill the single precision data arrays with artificial data

    float w = freqdata;
    float xy;
    for(j=0; j<ny; j++)  {
      for(i=0; i<nx; i++)  {
        xy = (float)(i*j)/(nx-1)*(ny-1);
//      Data for the internal FFT are separated
        if ( ffttype==FFT_INTERNAL ) {
          sarray1[i+j*2*nx] = cos(2*M_PI*w*xy);
          sarray1[i+nx+j*2*nx] = 0.0;
        }
//      Data for FFTW are interleaved
        if ( ffttype==FFT_FFTW ) {
          sarray1[2*i+j*2*nx] = cos(2*M_PI*w*xy);
          sarray1[2*i+1+j*2*nx] = 0.0;
        }
      }
    }
    memcpy(sarray2,sarray1,sizeof(sarray1)*fftdatasize);

// CPU code, single precision, timing loop

    if ( !gpuonly ) {

      if ( niter==1 && nx*ny<=64 ) {
        printf("CPU single before\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",sarray1[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      if ( ffttype==FFT_INTERNAL ) {
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          sfft (sarray2, sarray2+nx, nx, 1);
        }
        end = TIMER_FUNCTION;
      }
      else if ( ffttype==FFT_FFTW ) {

        pf = fftwf_plan_dft_2d(nx, ny, (fftwf_complex*) sarray1, (fftwf_complex*) sarray2,
              FFTW_FORWARD, FFTW_ESTIMATE);

        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          fftwf_execute(pf);
        }
        end = TIMER_FUNCTION;

        fftwf_destroy_plan(pf);

      }

      if ( niter==1 && nx*ny<=64 ) {
        printf("CPU single after\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",sarray2[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      *stimecpu = end-start;

      if ( niter==1 ) {
        // Find the frequency
        int freq = 0;
        for(j=0; j<ny/2; j++) {
          for(i=0; i<nx/2; i++) {
            if ( ffttype==FFT_INTERNAL ) {
              if( sarray2[i+j*2*nx] > sarray2[freq] )
                freq = i+j*2*nx;
            }
            if ( ffttype==FFT_FFTW ) {
              if( sarray2[2*i+j*2*nx] > sarray2[freq] )
                freq = 2*i+j*2*nx;
            }
          }
        }
        if ( ffttype==FFT_FFTW ) freq=freq/2;
        printf ("fft, CPU, single,                 frequency : %i\n",freq);
      }
    }

// GPU code, single precision, timing loop

    if ( !cpuonly ) {

//    Reset the input data for the next transform
      memcpy(sarray2,sarray1,sizeof(sarray1)*fftdatasize);

      if ( niter==1 && nx*ny<=64 ) {
        printf("GPU single before\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",sarray1[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      if ( ngpu==1 ) {
#ifdef USE_CUDA
        // Single GPU code

        // Query OpenACC for CUDA stream
        void *stream = acc_get_cuda_stream(acc_async_sync);
        sPlan2dCUFFT(nx, ny, stream);

        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
// Copy data to device at start of region and back to host and end of region
#pragma acc data copy(sarray2[0:fftdatasize])
// Inside this region the device data pointer will be used
#pragma acc host_data use_device(sarray2)
          {
            sExecCUFFT(sarray2);
          }
        }
        end = TIMER_FUNCTION;

        sDestroyCUFFT();
#endif
      }
      else {
#ifdef USE_CUDA
        // Multiple GPU code

        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          sMulti2dCUFFT(nx, ny, sarray2, ngpu);
        }
        end = TIMER_FUNCTION;
/*
        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
// Copy data to device at start of region and back to host and end of region
#pragma acc data copy(sarray2[0:fftdatasize])
// Inside this region the device data pointer will be used
#pragma acc host_data use_device(sarray2)
          {
            sMulti2dCUFFT(nx, ny, sarray2, ngpu);
          }
        }
        end = TIMER_FUNCTION;
*/
#endif
      }

      if ( niter==1 && nx*ny<=64 ) {
        printf("GPU single after\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",sarray2[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      *stimegpu = end-start;

      if ( niter==1 ) {
        // Find the frequency
        int freq = 0;
        for(j=0; j<ny/2; j++) {
          for(i=0; i<nx/2; i++) {
            if ( ffttype==FFT_INTERNAL ) {
              if( sarray2[i+j*2*nx] > sarray2[freq] )
                freq = i+j*2*nx;
            }
            if ( ffttype==FFT_FFTW ) {
              if( sarray2[2*i+j*2*nx] > sarray2[freq] )
                freq = 2*i+j*2*nx;
            }
          }
        }
        if ( ffttype==FFT_FFTW ) freq=freq/2;
        printf ("fft, GPU, single,                 frequency : %i\n",freq);
      }
    }

  }

// ------------------------------------------------------------------
// DOUBLE PRECISION
// ------------------------------------------------------------------

  if ( doubledata ) {

// Fill the double precision data arrays with artificial data

    double w = freqdata;
    double xy;
    for(j=0; j<ny; j++)  {
      for(i=0; i<nx; i++)  {
        xy = (double)(i*j)/(nx-1)*(ny-1);
//      Data for the internal FFT are separated
        if ( ffttype==FFT_INTERNAL ) {
          darray1[i+2*j*nx] = cos(2*M_PI*w*xy);
          darray1[i+nx+2*j*nx] = 0.0;
        }
//      Data for FFTW are interleaved
        if ( ffttype==FFT_FFTW ) {
          darray1[2*i+2*j*nx] = cos(2*M_PI*w*xy);
          darray1[2*i+1+2*j*nx] = 0.0;
        }
      }
    }
    memcpy(darray2,darray1,sizeof(darray1)*fftdatasize);

// CPU code, DOUBLE precision, timing loop

    if ( !gpuonly ) {

      if ( niter==1 && nx*ny<=64 ) {
        printf("CPU double before\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",darray1[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      if ( ffttype==FFT_INTERNAL ) {
        printf("No 2D version implemented for the internal FFT\n");
        return (1);
      }
      else if ( ffttype==FFT_FFTW ) {

        p = fftw_plan_dft_2d(nx, ny, (fftw_complex*) darray1, (fftw_complex*) darray2,
              FFTW_FORWARD, FFTW_ESTIMATE);

        start = TIMER_FUNCTION;
        for (iter=0;iter<niter;iter++ ) {
          fftw_execute(p);
        }
        end = TIMER_FUNCTION;

        fftw_destroy_plan(p);

      }

      if ( niter==1 && nx*ny<=64 ) {
        printf("CPU double after\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",darray2[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      *dtimecpu = end-start;

      if ( niter==1 ) {
        // Find the frequency
        int freq = 0;
        for(j=0; j<ny/2; j++) {
          for(i=0; i<nx/2; i++) {
            if ( ffttype==FFT_INTERNAL ) {
              if( darray2[i+j*2*nx] > darray2[freq] )
                freq = i+j*2*nx;
            }
            if ( ffttype==FFT_FFTW ) {
              if( darray2[2*i+j*2*nx] > darray2[freq] )
                freq = 2*i+j*2*nx;
            }
          }
        }
        if ( ffttype==FFT_FFTW ) freq=freq/2;
        printf ("fft, CPU, double,                 frequency : %i\n",freq);
      }
    }

// GPU code, DOUBLE precision, timing loop

    if ( !cpuonly ) {

#ifdef USE_CUDA
//    Reset the input data for the next transform
      memcpy(darray2,darray1,sizeof(darray1)*fftdatasize);

      if ( niter==1 && nx*ny<=64 ) {
        printf("GPU double before\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",darray1[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      // Query OpenACC for CUDA stream
      void *stream = acc_get_cuda_stream(acc_async_sync);
      dPlan2dCUFFT(nx, ny, stream);

      start = TIMER_FUNCTION;
      for (iter=0;iter<niter;iter++ ) {
// Copy data to device at start of region and back to host and end of region
#pragma acc data copy(darray2[0:fftdatasize])
// Inside this region the device data pointer will be used
#pragma acc host_data use_device(darray2)
        {
          dExecCUFFT(darray2);
        }
      }
      end = TIMER_FUNCTION;

      dDestroyCUFFT();

      if ( niter==1 && nx*ny<=64 ) {
        printf("GPU double after\n");
        for (i=0;i<fftdatasize;i++) {
          printf("%6.3f ",darray2[i]);
          if ( (i+1)%8==0 ) printf("\n");
        }
      }

      *dtimegpu = end-start;

      if ( niter==1 ) {
        // Find the frequency
        int freq = 0;
        for(j=0; j<ny/2; j++) {
          for(i=0; i<nx/2; i++) {
            if ( ffttype==FFT_INTERNAL ) {
              if( darray2[i+j*2*nx] > darray2[freq] )
                freq = i+j*2*nx;
            }
            if ( ffttype==FFT_FFTW ) {
              if( darray2[2*i+j*2*nx] > darray2[freq] )
                freq = 2*i+j*2*nx;
            }
          }
        }
        if ( ffttype==FFT_FFTW ) freq=freq/2;
        printf ("fft, GPU, double,                 frequency : %i\n",freq);
      }
#endif
    }

  }

  return ierr;

}

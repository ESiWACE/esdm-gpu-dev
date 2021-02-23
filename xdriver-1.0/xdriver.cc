#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cfloat>          // for DBL_MAX
#include <math.h>
#include <time.h>
#include <unistd.h>        // for gethostname
#if defined(_OPENMP)
#include <omp.h>
#define TIMER_FUNCTION omp_get_wtime()
#else
#define TIMER_FUNCTION (double)clock()/CLOCKS_PER_SEC
#endif
#define __MAIN__
#define STR_MAX_FILENAME 80
#define STR_MAX_TEXT 80

#include "compare.h"
#include "functions.h"

int test_fft1d (float *sarray, double *darray, int &nx, int &ny, int &niter,
         int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
         int &do_ffttype,
         double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu);

int test_fft2d (float *sarray, double *darray, int &nx, int &ny, int &niter,
         int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
         int &do_ffttype,
         double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu);

int test_varrayMinMaxMV (float *sarray, double *darray, int &nx, int &ny, int &niter,
         int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
         int &do_isnan, int &do_missval,
         double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu);

int test_vert_interp_lev (float *sarray, double *darray, int &nx, int &ny, int &nlev, int &niter,
         int &cpuonly, int &gpuonly, int &singledata, int &doubledata,
         int &do_isnan, int &do_missval,
         double *stimecpu, double *stimegpu, double *dtimecpu, double *dtimegpu);

void timing_report (char routine[]);

// Variables with global scope
int nlev, nx, ny;
int timingdata;
int cpuonly, gpuonly, singledata, doubledata;
int do_ffttype, do_isnan, do_missval;
double dtimecpu, dtimegpu, stimecpu, stimegpu;
FILE *file_handle;

int main (int argc, char *argv[]) {

// Local variables
  int errcount, i, ierr, irun, narray, niter, nrun, safety;
//int debug;
  int errlimit, help, restart, timingrun, unknown;
  int size, sizemax;
  int *nlevrun, *nxrun, *nyrun;
  int do_fft1d, do_fft2d, do_varrayMinMaxMV, do_vert_interp_lev;
  double memusage;
  double *darray;
  float  *sarray;
  char routine[STR_MAX_TEXT], runstamp[STR_MAX_TEXT], text[STR_MAX_TEXT];
  char datafn[STR_MAX_FILENAME], restartfn[STR_MAX_FILENAME], timingfn[STR_MAX_FILENAME];

  memset(routine, '\0', sizeof(routine));
  memset(text, '\0', sizeof(text));
  memset(datafn, '\0', sizeof(datafn));
  memset(restartfn, '\0', sizeof(restartfn));
  memset(runstamp, '\0', sizeof(restartfn));
  memset(timingfn, '\0', sizeof(timingfn));

  printf("=========================================================================");
  printf ("Timing harness for CDO operators\n");

// Read steering data from a restart file if present

  restart = 1;
  strcpy(restartfn,".xdriverrc");
  file_handle = fopen( restartfn, "r");
  if(file_handle==NULL) {
    // printf("xdriver: restart file %s does not exist\n",restartfn);
    restart = 0;
  }
  else {
    ierr = fscanf(file_handle,"%i",&nx);
    // printf("xdriver: options read from restart file %i\n",nx);
    if ( ierr==EOF ) {
      printf("xdriver: end of file reached reading from %s\n",restartfn);
    }
    else {
      restart = 0;
    }
    fclose(file_handle);
  }

// Set default options and settings. Reminder false=0; true=1

  nlev = 1;
  nx = 1000;
  ny = 500;
  doubledata=1;
  singledata=0;

  cpuonly=0;
  gpuonly=0;
/*
#ifdef DEBUG
  debug=1;
#else
  debug=0;
#endif
*/
  errcount=0;
  errlimit=0;
  niter=100;
  nrun=0;
  help=0;
  timingrun=0;
  do_isnan=1;
  do_missval=1;
  do_fft1d=0;
  do_fft2d=0;
  do_ffttype=FFT_FFTW;
  do_varrayMinMaxMV=0;
  do_vert_interp_lev=0;
  memset(text, '\0', sizeof(text));
  memset(timingfn, '\0', sizeof(timingfn));

// Make a stamp for the data output with system name, date and time

  ierr = gethostname(runstamp,STR_MAX_TEXT);
  time_t current_time = time(NULL);
  struct tm *tm = localtime(&current_time);
  i = strlen(runstamp);
  runstamp[i] = ' ';
  strftime(runstamp+i+1, sizeof(runstamp)-1, "%c", tm);

// Announce the program

  printf("Running xdriver on system %s\n",runstamp);
  printf("=========================================================================");

// Parse command arguments

  for (i=1;i<argc;i++) {
    unknown=1;
    if ( (strlen(argv[i])==2 && !strcmp(argv[i],"-h")) ||
         (strlen(argv[i])==5 && !strcmp(argv[i],"-help")) ) {
      help=1;
      unknown=0;
    }
    if ( strlen(argv[i])==8 && !strcmp(argv[i],"-cpuonly") ) {
      cpuonly=1;
      unknown=0;
    }
    if ( strlen(argv[i])==8 && !strcmp(argv[i],"-gpuonly") ) {
      gpuonly=1;
      unknown=0;
    }
    if ( strlen(argv[i])==7 && !strcmp(argv[i],"-double") ) {
      doubledata=1;
      unknown=0;
    }
    if ( strlen(argv[i])==strlen("-nodouble") && !strcmp(argv[i],"-nodouble") ) {
      doubledata=0;
      unknown=0;
    }
    if ( strlen(argv[i])==9 && !strcmp(argv[i],"-errlimit") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-errlimit\" parameter: error limit missing\n");
        errcount++;
      }
      else {
        i++;
        ierr = sscanf(argv[i],"%i",&errlimit);
        if ( ierr!=1 ) {
          printf("Error in \"-errlimit\" parameter: invalid number %s\n",argv[i]);
          errcount++;
        }
      }
      unknown=0;
    }
    if ( strlen(argv[i])==4 && !strcmp(argv[i],"-fft") ||
         strlen(argv[i])==6 && !strcmp(argv[i],"-fft1d") ) {
      do_fft1d=1;
      unknown=0;
    }
    if ( strlen(argv[i])==6 && !strcmp(argv[i],"-fft2d") ) {
      do_fft2d=1;
      unknown=0;
    }
    if ( strlen(argv[i])==5 && !strcmp(argv[i],"-ffti") ||
         strlen(argv[i])==12 && !strcmp(argv[i],"-fftinternal") ) {
      do_ffttype=FFT_INTERNAL;
      unknown=0;
    }
    if ( strlen(argv[i])==5 && !strcmp(argv[i],"-fftw") ) {
      do_ffttype=FFT_FFTW;
      unknown=0;
    }
    if ( strlen(argv[i])==6 && !strcmp(argv[i],"-niter") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-niter\" parameter: number of iterations missing\n");
        errcount++;
      }
      else {
        i++;
        ierr = sscanf(argv[i],"%i",&niter);
        if ( ierr!=1 ) {
          printf("Error in \"-niter\" parameter: invalid number %s\n",argv[i]);
          errcount++;
        }
      }
      unknown=0;
    }
    if ( strlen(argv[i])==5 && !strcmp(argv[i],"-nlev") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-nlev\" parameter: number of levels missing\n");
        errcount++;
      }
      else {
        i++;
        ierr = sscanf(argv[i],"%i",&nlev);
        if ( ierr!=1 ) {
          printf("Error in \"-nlev\" parameter: invalid number %s\n",argv[i]);
          errcount++;
        }
      }
      unknown=0;
    }
    if ( strlen(argv[i])==3 && !strcmp(argv[i],"-nx") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-nx\" parameter: x dimension missing\n");
        errcount++;
      }
      else {
        i++;
        ierr = sscanf(argv[i],"%i",&nx);
        if ( ierr!=1 ) {
          printf("Error in \"-nx\" parameter: invalid number %s\n",argv[i]);
          errcount++;
        }
      }
      unknown=0;
    }
    if ( strlen(argv[i])==3 && !strcmp(argv[i],"-ny") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-ny\" parameter: y dimension missing\n");
        errcount++;
      }
      else {
        i++;
        ierr = sscanf(argv[i],"%i",&ny);
        if ( ierr!=1 ) {
          printf("Error in \"-ny\" parameter: invalid number %s\n",argv[i]);
          errcount++;
        }
      }
      unknown=0;
    }
    if ( strlen(argv[i])==6 && !strcmp(argv[i],"-isnan") ) {
      do_isnan=1;
      unknown=0;
    }
    if ( strlen(argv[i])==8 && !strcmp(argv[i],"-noisnan") ) {
      do_isnan=0;
      unknown=0;
    }
    if ( strlen(argv[i])==8 && !strcmp(argv[i],"-missval") ) {
      do_missval=1;
      unknown=0;
    }
    if ( strlen(argv[i])==10 && !strcmp(argv[i],"-nomissval") ) {
      do_missval=0;
      unknown=0;
    }
    if ( strlen(argv[i])==7 && !strcmp(argv[i],"-single") ) {
      singledata=1;
      unknown=0;
    }
    if ( strlen(argv[i])==strlen("-nosingle") && !strcmp(argv[i],"-nosingle") ) {
      singledata=0;
      unknown=0;
    }
    if ( strlen(argv[i])==2 && !strcmp(argv[i],"-t") ||
         strlen(argv[i])==7 && !strcmp(argv[i],"-timing") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-timing\" parameter: timing file name missing\n");
        errcount++;
      }
      else {
        i++;
        strcpy(timingfn,argv[i]);
        timingrun=1;
      }
      unknown=0;
    }
    if ( strlen(argv[i])==3 && !strcmp(argv[i],"-to") ||
         strlen(argv[i])==14 && !strcmp(argv[i],"-timing_output") ) {
      if ( i+1>argc-1 ) {
        printf("Error in \"-timing_output\" parameter: timing output file name missing\n");
        errcount++;
      }
      else {
        i++;
        strcpy(datafn,argv[i]);
        timingrun=1;
      }
      unknown=0;
    }
    if ( strlen(argv[i])==4 && !strcmp(argv[i],"-vmm") ||
         strlen(argv[i])==15 && !strcmp(argv[i],"-varrayMinMaxMV") ) {
      do_varrayMinMaxMV=1;
      unknown=0;
    }
    if ( strlen(argv[i])==5 && !strcmp(argv[i],"-vert") ||
         strlen(argv[i])==16 && !strcmp(argv[i],"-vert_interp_lev") ) {
      do_vert_interp_lev=1;
      unknown=0;
    }
    if ( unknown ) {
      printf("Error: argument unrecognised %s\n",argv[i]);
      errcount++;
    }
  } // loop over argc

// Print help information

  if ( help ) {
    printf("usage: \"xdriver <operators> <options>\"\n");
    printf("operators:\n");
    printf("   -fft                   FFT kernel\n");
    printf("   -vmm                   varrayMinMaxMV\n");
    printf("   -vert                  vert_interp_lev\n");
    printf("options:\n");
    printf("   -nx                    Size of x dimension of data arrays\n");
    printf("   -nx                    Size of x dimension of data arrays\n");
    printf("   -nx                    Size of x dimension of data arrays\n");
    printf("   -ny                    Size of y dimension of data arrays\n");
    printf("   -cpuonly               Run using CPU code not on the GPU\n");
    printf("   -gpuonly               Run using GPU code not on the CPU\n");
    printf("   -double                Run with double precision data (default)\n");
    printf("   -nodouble              Do not run with double precision data\n");
    printf("   -errlimit <n>          Maximum number of erroneous values to print\n");
    printf("   -isnan                 Use the equality test which checks for NaNs (default)\n");
    printf("   -noisnan               Do not check for NaNs\n");
    printf("   -missval               Check data for missing values (default)\n");
    printf("   -nomissval             Do not check data for missing values\n");
    printf("   -niter <n>             Number of iterations for timing\n");
    printf("   -single                Run with single precision data\n");
    printf("   -nosingle              Do not run with single precision data (default)\n");
    printf("   -timing <file>         File containing list of data sizes for timing run\n");
    printf("   -timing_output <file>  Output file for data from the timing run\n");
    printf("   -help                  Print this help information\n");
    exit(1);
  }

// Exit if there has been an error in parsing the arg list

  if ( errcount ) {
    exit(1);
  }

  printf ("xdriver: %i iterations for timing\n",niter);

// Save options in a restart file

  if ( restart ) { 
    file_handle = fopen( restartfn, "w");
    if(file_handle==NULL) {
    }
    else {
      ierr = fprintf(file_handle,"%i %i %i\n",nx,ny,nlev);
      if ( ierr!=1 ) {
        printf("xdriver: error code %i unable to write to restart file %s\n",ierr,restartfn);
      }
      ierr = fwrite("\n",1,1,file_handle);
    }
    fclose(file_handle);
  }

// Read a file of sizes for a timingrun

  if ( timingrun ) {

    file_handle = fopen( timingfn, "r");
    if(file_handle==NULL) {
      printf("xdriver: unable to read from timing file %s\n",timingfn);
      exit (1);
    }
    else {

//    Read the number of entries
      ierr = fscanf(file_handle,"%i",&nrun);
      if ( ierr==EOF ) {
        printf("xdriver: end of file reached reading from %s\n",timingfn);
        exit (1);
      }
      if ( ierr!=1 ) {
        printf("xdriver: invalid input reading from %s\n",timingfn);
        exit (1);
      }
      if ( nrun<=0 ) {
        printf("xdriver: invalid nrun %i read from file %s\n",nrun,timingfn);
        exit (1);
      }
      else {
        int int1, int2, int3;
        // char str1[STR_MAX_TEXT];
        nlevrun = (int *)malloc(nrun*sizeof(int));
        nxrun = (int *)malloc(nrun*sizeof(int));
        nyrun = (int *)malloc(nrun*sizeof(int));
        for (irun=0;irun<nrun;irun++) {
          ierr = fscanf(file_handle,"%i %i %i",&int1,&int2,&int3);
          if ( ierr==EOF ) {
            printf("xdriver: end of file reached reading from %s\n",timingfn);
            exit (1);
          }
          if ( ierr!=3 ) {
            printf("xdriver: invalid input reading from %s\n",timingfn);
            exit (1);
          }
          *(nlevrun+irun) = int1;
          *(nxrun+irun) = int2;
          *(nyrun+irun) = int3;
        }
      }

//    Read the data file name, if present
      ierr = fscanf(file_handle,"%s",text);
      if ( ierr==EOF ) {
        printf("xdriver: warning end of file reached reading from %s\n",timingfn);
        exit (1);
      }
      if ( ierr!=1 ) {
        printf("xdriver: invalid input reading from %s\n",timingfn);
        exit (1);
      }
//    Only overwrite the current value if not set from command line
      if ( !strcmp(datafn,"") ) strcpy(datafn,text);
      fclose(file_handle);
    }
    file_handle = fopen( datafn, "r");
    if(file_handle!=NULL) {
      printf("xdriver: ERROR timing data file %s already exists\n",datafn);
      exit(1);
    }
    file_handle = fopen( datafn, "w");
    if(file_handle==NULL) {
      timingdata = 0;
    }
    else {
      timingdata = 1;
      fprintf(file_handle,"%s\n",runstamp);
    }
  } // timingrun
  else {

// No timing run, set values for a single run

    nrun = 1;
    nlevrun = (int *)malloc(nrun*sizeof(int));
    nxrun = (int *)malloc(nrun*sizeof(int));
    nyrun = (int *)malloc(nrun*sizeof(int));
    *(nlevrun) = nlev;
    *(nxrun) = nx;
    *(nyrun) = ny;
  }

// Find the maximum data sizes for array allocation

  sizemax = 0;
  for (irun=0;irun<nrun;irun++) {
    size = (*(nlevrun+irun))*(*(nxrun+irun))*(*(nyrun+irun));
    if ( size>sizemax ) sizemax = size;
  }

// Allocate the main data arrays

  safety = 2;
  narray = 1;
  if ( do_vert_interp_lev ) narray = 2;
// For FFTs we need two arrays of complex data
  if ( do_fft1d || do_fft2d ) narray = 4;
  printf ("xdriver: allocating %i arrays of size %i\n",narray,sizemax);
  if ( doubledata ) {
    memusage = safety*narray*sizemax*sizeof(double)/(1024.0*1024.0);
    darray = (double*)malloc(safety*narray*sizemax*sizeof(double));
    if ( darray == NULL ) {
      printf("Error: malloc failed to find %f MB\n",memusage);
      exit(1);
    }
    // Share memory for the double and single precision arrays
    sarray = (float *) darray;
  }
  else if ( singledata ) {
    memusage = safety*narray*sizemax*sizeof(float)/(1024.0*1024.0);
    sarray = (float*)malloc(safety*narray*sizemax*sizeof(float));
    if ( sarray == NULL ) {
      printf("Error: malloc failed to find %f MB\n",memusage);
      exit(1);
    }
  }
  else {
    printf("Error: neither single nor double precision selected\n");
    exit(1);
  }
  printf ("xdriver: memory usage %f MB\n",memusage);

  for (irun=0;irun<nrun;irun++) {

    if ( timingrun ) {
      nlev = *(nlevrun+irun);
      nx = *(nxrun+irun);
      ny = *(nyrun+irun);
    }
    printf ("xdriver: run %i array sizes %i x %i x %i\n",irun,nlev,nx,ny);

//   Run tests as per the do_* flags

    if ( !do_fft1d && !do_fft2d && !do_varrayMinMaxMV && !do_vert_interp_lev ) {
      printf("WARNING: no operator selected, see -help\n");
    }

    if ( do_fft1d ) {
      ierr = test_fft1d(sarray, darray, nx, ny, niter,
          cpuonly, gpuonly, singledata, doubledata,
          do_ffttype,
          &stimecpu,&stimegpu,&dtimecpu,&dtimegpu);
      if ( ierr ) {
        printf ("xdriver: test_fft1d failed due to errors, ierr = %i\n",ierr);
        exit (1);
      }
      strcpy(routine,"fft1d");
      timing_report(routine);
    }

    if ( do_fft2d ) {
      ierr = test_fft2d(sarray, darray, nx, ny, niter,
          cpuonly, gpuonly, singledata, doubledata,
          do_ffttype,
          &stimecpu,&stimegpu,&dtimecpu,&dtimegpu);
      if ( ierr ) {
        printf ("xdriver: test_fft2d failed due to errors, ierr = %i\n",ierr);
        exit (1);
      }
      strcpy(routine,"fft2d");
      timing_report(routine);
    }

    if ( do_varrayMinMaxMV ) {
      ierr = test_varrayMinMaxMV(sarray, darray, nx, ny, niter,
          cpuonly, gpuonly, singledata, doubledata,
          do_isnan, do_missval,
          &stimecpu,&stimegpu,&dtimecpu,&dtimegpu);
      if ( ierr ) {
        printf ("xdriver: test_varrayMinMaxMV failed due to errors, ierr = %i\n",ierr);
        exit (1);
      }
      strcpy(routine,"varrayMinMaxMV");
      timing_report(routine);
    }

    if ( do_vert_interp_lev ) {
      ierr = test_vert_interp_lev(sarray, darray, nx, ny, nlev, niter,
          cpuonly, gpuonly, singledata, doubledata,
          do_isnan, do_missval,
          &stimecpu,&stimegpu,&dtimecpu,&dtimegpu);
      if ( ierr ) {
        printf ("xdriver: test_vert_interp_lev failed due to errors, ierr = %i\n",ierr);
        exit (1);
      }
      strcpy(routine,"vert_interp_lev");
      timing_report(routine);
    }
  }

}

void timing_report (char routine[]) {

  char ctext[32];
  char gtext[32];
// Timing report

  if ( do_isnan && do_missval ) {
    strcpy(ctext,"with    isnan, with    missval");
  }
  else if ( do_missval ) {
    strcpy(ctext,"without isnan, with    missval");
  }
  else {
    strcpy(ctext,"without isnan, without missval");
  }
  strcpy(gtext,ctext);
  if ( !strcmp(routine,"fft1d") || !strcmp(routine,"fft2d") ) {
    if ( do_ffttype==FFT_FFTW ) {
      strcpy(ctext,"using FFTW                    ");
    }
    if ( do_ffttype==FFT_INTERNAL ) {
      strcpy(ctext,"using internal FFT            ");
    }
    strcpy(gtext,"using CUFFT                   ");
  }

  if ( singledata && !gpuonly ) 
    printf ("%s, CPU, single, %s:  %f secs\n",routine,ctext,stimecpu);
  if ( singledata && !cpuonly ) 
    printf ("%s, GPU, single, %s:  %f secs\n",routine,gtext,stimegpu);
  if ( doubledata && !gpuonly ) 
    printf ("%s, CPU, double, %s:  %f secs\n",routine,ctext,dtimecpu);
  if ( doubledata && !cpuonly ) 
    printf ("%s, GPU, double, %s:  %f secs\n",routine,gtext,dtimegpu);

  if ( timingdata ) {
    fprintf(file_handle,"%s %i %i %i %f %f %f %f\n",routine,nlev,nx,ny,
      stimecpu,stimegpu,dtimecpu,dtimegpu);
  }
}


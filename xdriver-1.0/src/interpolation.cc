#include <cstddef>
#include <iostream>
#include "compare.h"

static float
s_vert_interp_lev_kernel(float w1, float w2, float var1L1, float var1L2, float missval)
{
  float var2 = missval;

  if (DBL_IS_EQUAL(var1L1, missval)) w1 = 0;
  if (DBL_IS_EQUAL(var1L2, missval)) w2 = 0;

  if (IS_EQUAL(w1, 0) && IS_EQUAL(w2, 0))
    {
      var2 = missval;
    }
  else if (IS_EQUAL(w1, 0))
    {
      var2 = (w2 >= 0.5) ? var1L2 : missval;
    }
  else if (IS_EQUAL(w2, 0))
    {
      var2 = (w1 >= 0.5) ? var1L1 : missval;
    }
  else
    {
      var2 = var1L1 * w1 + var1L2 * w2;
    }

  return var2;
}

static float
s_vert_interp_lev_kernel_noisnan(float w1, float w2, float var1L1, float var1L2, float missval)
{
  float var2 = missval;

  if (IS_EQUAL(var1L1, missval)) w1 = 0;
  if (IS_EQUAL(var1L2, missval)) w2 = 0;

  if (IS_EQUAL(w1, 0) && IS_EQUAL(w2, 0))
    {
      var2 = missval;
    }
  else if (IS_EQUAL(w1, 0))
    {
      var2 = (w2 >= 0.5) ? var1L2 : missval;
    }
  else if (IS_EQUAL(w2, 0))
    {
      var2 = (w1 >= 0.5) ? var1L1 : missval;
    }
  else
    {
      var2 = var1L1 * w1 + var1L2 * w2;
    }

  return var2;
}

static float
s_vert_interp_lev_kernel_nomissval(float w1, float w2, float var1L1, float var1L2, float missval)
{
  return var1L1 * w1 + var1L2 * w2;
}

static double
d_vert_interp_lev_kernel(double w1, double w2, double var1L1, double var1L2, double missval)
{
  double var2 = missval;

  if (DBL_IS_EQUAL(var1L1, missval)) w1 = 0;
  if (DBL_IS_EQUAL(var1L2, missval)) w2 = 0;

  if (IS_EQUAL(w1, 0) && IS_EQUAL(w2, 0))
    {
      var2 = missval;
    }
  else if (IS_EQUAL(w1, 0))
    {
      var2 = (w2 >= 0.5) ? var1L2 : missval;
    }
  else if (IS_EQUAL(w2, 0))
    {
      var2 = (w1 >= 0.5) ? var1L1 : missval;
    }
  else
    {
      var2 = var1L1 * w1 + var1L2 * w2;
    }

  return var2;
}

static double
d_vert_interp_lev_kernel_noisnan(double w1, double w2, double var1L1, double var1L2, double missval)
{
  double var2 = missval;

  if (IS_EQUAL(var1L1, missval)) w1 = 0;
  if (IS_EQUAL(var1L2, missval)) w2 = 0;

  if (IS_EQUAL(w1, 0) && IS_EQUAL(w2, 0))
    {
      var2 = missval;
    }
  else if (IS_EQUAL(w1, 0))
    {
      var2 = (w2 >= 0.5) ? var1L2 : missval;
    }
  else if (IS_EQUAL(w2, 0))
    {
      var2 = (w1 >= 0.5) ? var1L1 : missval;
    }
  else
    {
      var2 = var1L1 * w1 + var1L2 * w2;
    }

  return var2;
}

static double
d_vert_interp_lev_kernel_nomissval(double w1, double w2, double var1L1, double var1L2, double missval)
{
  return var1L1 * w1 + var1L2 * w2;
}

void
CPUs_vert_interp_lev(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      float *var2 = vardata2 + gridsize * ilev;
      const float *var1L1 = vardata1 + gridsize * idx1;
      const float *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = s_vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
CPUs_vert_interp_lev_noisnan(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      float *var2 = vardata2 + gridsize * ilev;
      const float *var1L1 = vardata1 + gridsize * idx1;
      const float *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = s_vert_interp_lev_kernel_noisnan(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
CPUs_vert_interp_lev_nomissval(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      float *var2 = vardata2 + gridsize * ilev;
      const float *var1L1 = vardata1 + gridsize * idx1;
      const float *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = s_vert_interp_lev_kernel_nomissval(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
GPUs_vert_interp_lev(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
/*
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      for (size_t i = 0; i < gridsize; ++i)
        {
printf("GPU before %f %f\n",*(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i));
        }
    }
*/
#pragma acc data copyin(vardata1[:gridsize*nlev2])
#pragma acc data copyout(vardata2[:gridsize*nlev2])
#pragma acc parallel loop collapse(2)
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          *(vardata2+gridsize*ilev+i) = s_vert_interp_lev_kernel(lev_wgt1[ilev], lev_wgt2[ilev], 
          *(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i), missval);
        }
    }
/*
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      for (size_t i = 0; i < gridsize; ++i)
        {
printf("GPU after  %f\n", *(vardata2+gridsize*ilev+i));
        }
    }
*/
}

void
CPUd_vert_interp_lev_original(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;
      const double *var1L1 = vardata1 + gridsize * idx1;
      const double *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = d_vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
CPUd_vert_interp_lev_noisnan(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;
      const double *var1L1 = vardata1 + gridsize * idx1;
      const double *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = d_vert_interp_lev_kernel_noisnan(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
CPUd_vert_interp_lev_nomissval(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;
      const double *var1L1 = vardata1 + gridsize * idx1;
      const double *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = d_vert_interp_lev_kernel_nomissval(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
CPUd_vert_interp_lev(const size_t gridsize, const double missval,
 	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
/*
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      for (size_t i = 0; i < gridsize; ++i)
        {
printf("CPU before %f %f\n",*(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i));
        }
    }
*/
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;
      const double *var1L1 = vardata1 + gridsize * idx1;
      const double *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = d_vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
/*
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      for (size_t i = 0; i < gridsize; ++i)
        {
printf("CPU after  %f\n", *(vardata2+gridsize*ilev+i));
        }
    }
*/
}

void
GPUs_vert_interp_lev_original(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      float *var2 = vardata2 + gridsize * ilev;
      const float *var1L1 = vardata1 + gridsize * idx1;
      const float *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
#pragma acc parallel loop
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = s_vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
GPUs_vert_interp_lev_noisnan(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
#pragma acc data copyin(vardata1[:gridsize*nlev2])
#pragma acc data copyout(vardata2[:gridsize*nlev2])
#pragma acc parallel loop collapse(2)
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          *(vardata2+gridsize*ilev+i) = s_vert_interp_lev_kernel_noisnan(lev_wgt1[ilev], lev_wgt2[ilev], 
          *(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i), missval);
        }
    }
}

void
GPUs_vert_interp_lev_nomissval(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const float *lev_wgt1, const float *lev_wgt2)
{
#pragma acc data copyin(vardata1[:gridsize*nlev2])
#pragma acc data copyout(vardata2[:gridsize*nlev2])
#pragma acc parallel loop collapse(2)
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          *(vardata2+gridsize*ilev+i) = s_vert_interp_lev_kernel_nomissval(lev_wgt1[ilev], lev_wgt2[ilev], 
          *(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i), missval);
        }
    }
}

void
GPUd_vert_interp_lev_original(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;
      const double *var1L1 = vardata1 + gridsize * idx1;
      const double *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
#pragma acc parallel loop
      for (size_t i = 0; i < gridsize; ++i)
        {
          var2[i] = d_vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i], missval);
        }
    }
}

void
GPUd_vert_interp_lev_noisnan(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
#pragma acc data copyin(vardata1[:gridsize*nlev2])
#pragma acc data copyout(vardata2[:gridsize*nlev2])
#pragma acc parallel loop collapse(2)
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          *(vardata2+gridsize*ilev+i) = d_vert_interp_lev_kernel_noisnan(lev_wgt1[ilev], lev_wgt2[ilev], 
          *(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i), missval);
        }
    }
}

void
GPUd_vert_interp_lev_nomissval(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
#pragma acc data copyin(vardata1[:gridsize*nlev2])
#pragma acc data copyout(vardata2[:gridsize*nlev2])
#pragma acc parallel loop collapse(2)
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          *(vardata2+gridsize*ilev+i) = d_vert_interp_lev_kernel_nomissval(lev_wgt1[ilev], lev_wgt2[ilev], 
          *(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i), missval);
        }
    }
}

void
GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
/*
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      for (size_t i = 0; i < gridsize; ++i)
        {
printf("GPU before %f %f\n",*(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i));
        }
    }
*/
#pragma acc data copyin(vardata1[:gridsize*nlev2])
#pragma acc data copyout(vardata2[:gridsize*nlev2])
#pragma acc parallel loop collapse(2)
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
      for (size_t i = 0; i < gridsize; ++i)
        {
          *(vardata2+gridsize*ilev+i) = d_vert_interp_lev_kernel(lev_wgt1[ilev], lev_wgt2[ilev], 
          *(vardata1+gridsize*lev_idx1[ilev]+i), *(vardata1+gridsize*lev_idx2[ilev]+i), missval);
        }
    }
/*
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      for (size_t i = 0; i < gridsize; ++i)
        {
printf("GPU after  %f\n", *(vardata2+gridsize*ilev+i));
        }
    }
*/
}


/*

#define MAXGRID 10000
#define MAXLEV 8

void
tiled_GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
  if ( nlev2>MAXLEV ) {
    printf("ERROR in GPUd_vert_interp_lev number of levels %i exceeds design %i\n",nlev2,MAXGRID);
    exit (1);
  }

  int tilesize = MAXGRID;
  for (int istart=0;istart<gridsize;istart=istart+MAXGRID) {
    if ( istart+MAXGRID>gridsize ) tilesize = gridsize-istart;
    printf("executing tile starting %i of size %i\n",istart,tilesize);

#pragma acc data copyin(vardata1[istart:MAXGRID*MAXLEV],vardata2[istart:MAXGRID*MAXLEV])
#pragma acc data copyin(lev_idx1[:MAXLEV],lev_idx2[:MAXLEV],lev_wgt1[:MAXLEV],lev_wgt2[:MAXLEV])
#pragma acc parallel loop
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      double wgt1 = lev_wgt1[ilev];
      double wgt2 = lev_wgt2[ilev];
      const int lev1 = lev_idx1[ilev];
      const int lev2 = lev_idx2[ilev];
#pragma acc data copyin (vardata1[MAXGRID*lev1+istart:MAXGRID*lev1+MAXGRID]) 
#pragma acc data copyin (vardata2[MAXGRID*lev2+istart:MAXGRID*lev2+MAXGRID]) 
#pragma acc data copyout(vardata2[MAXGRID*ilev+istart:MAXGRID*ilev+MAXGRID]) 
#pragma acc loop independent
      for (size_t i = istart; i < istart+tilesize; ++i)
        {
          if (vardata1[i+gridsize*lev1]==missval) wgt1 = 0;
          if (vardata2[i+gridsize*lev2]==missval) wgt2 = 0;

          if ((wgt1==0) && (wgt2==0))
            {
              vardata2[i+gridsize*ilev] = missval;
            }
          else if (wgt1==0)
            {
              vardata2[i+gridsize*ilev] = (wgt2 >= 0.5) ? vardata2[i+gridsize*lev2] : missval;
            }
          else if (wgt2==0)
            {
              vardata2[i+gridsize*ilev] = (wgt1 >= 0.5) ? vardata1[i+gridsize*lev1] : missval;
            }
          else
            {
              vardata2[i+gridsize*ilev] = vardata1[i+gridsize*lev1] * wgt1 
                                        + vardata2[i+gridsize*lev2] * wgt2;
            }
        }
    }
  }
}

void
mask_GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, const int *lev_idx1, const int *lev_idx2,
                const double *lev_wgt1, const double *lev_wgt2)
{
//#pragma acc data copyin(vardata1[:gridsize*nlev2],vardata2[:gridsize*nlev2])
//#pragma acc data copyin(lev_idx1[:nlev2],lev_idx2[:nlev2],lev_wgt1[:nlev2],lev_wgt2[:nlev2])
  
//#pragma acc parallel loop
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      double wgt1 = lev_wgt1[ilev];
      double wgt2 = lev_wgt2[ilev];
      const int lev1 = lev_idx1[ilev];
      const int lev2 = lev_idx2[ilev];
#pragma acc data copyin(vardata1[gridsize*lev1:gridsize*lev1+gridsize]) 
#pragma acc data copyin(vardata2[gridsize*lev2:gridsize*lev2+gridsize]) 
#pragma acc data copyout(vardata2[gridsize*ilev:gridsize*ilev+gridsize]) 

#pragma acc loop independent
      for (size_t i = 0; i < gridsize; ++i) {
        int wm = ((vardata1[i+gridsize*lev1]==missval) && (vardata2[i+gridsize*lev2]==missval))
                 || ((vardata1[i+gridsize*lev1]==missval) && (vardata2[i+gridsize*lev2]!=missval)
                      && wgt2<0.5)
                 || ((vardata1[i+gridsize*lev1]!=missval) && (vardata2[i+gridsize*lev2]==missval)
                      && wgt1<0.5);
        int w1 = ((vardata1[i+gridsize*lev1]!=missval) && (vardata2[i+gridsize*lev2]==missval)
                      && wgt1>=0.5);
        int w2 = ((vardata1[i+gridsize*lev1]==missval) && (vardata2[i+gridsize*lev2]!=missval)
                      && wgt2>=0.5);
        int wb = ((vardata1[i+gridsize*lev1]!=missval) && (vardata2[i+gridsize*lev2]!=missval));

        if ( wm ) vardata2[i+gridsize*ilev] = missval;
        if ( w1 ) vardata2[i+gridsize*ilev] = vardata1[i+gridsize*lev1];
        if ( w2 ) vardata2[i+gridsize*ilev] = vardata2[i+gridsize*lev2];
        if ( wb ) vardata2[i+gridsize*ilev] = 
                      (vardata1[i+gridsize*lev1]*wgt1 + vardata2[i+gridsize*lev2]*wgt2);
        

// Original code or a version of it

          if (vardata1[i+gridsize*lev1]==missval) wgt1 = 0;
          if (vardata2[i+gridsize*lev2]==missval) wgt2 = 0;

          if ((wgt1==0) && (wgt2==0))
            {
              vardata2[i+gridsize*ilev] = missval;
            }
          else if (wgt1==0)
            {
              vardata2[i+gridsize*ilev] = (wgt2 >= 0.5) ? vardata2[i+gridsize*lev2] : missval;
            }
          else if (wgt2==0)
            {
              vardata2[i+gridsize*ilev] = (wgt1 >= 0.5) ? vardata1[i+gridsize*lev1] : missval;
            }
          else
            {
              vardata2[i+gridsize*ilev] = vardata1[i+gridsize*lev1] * wgt1 
                                        + vardata2[i+gridsize*lev2] * wgt2;
            }
//
        }
    }
}

*/

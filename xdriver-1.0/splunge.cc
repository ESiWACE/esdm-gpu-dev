#include <cstddef>
#include "compare.h"

static float
s_vert_interp_lev_kernel(float w1, float w2, float var1L1, float var1L2, float missval)
{
  float var2 = missval;

//if (DBL_IS_EQUAL(var1L1, missval)) w1 = 0;
//if (DBL_IS_EQUAL(var1L2, missval)) w2 = 0;
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
d_vert_interp_lev_kernel(double w1, double w2, double var1L1, double var1L2, double missval)
{
  double var2 = missval;

//if (DBL_IS_EQUAL(var1L1, missval)) w1 = 0;
//if (DBL_IS_EQUAL(var1L2, missval)) w2 = 0;
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

void
CPUs_vert_interp_lev(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, int *lev_idx1, int *lev_idx2,
                float *lev_wgt1, float *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      float *var2 = vardata2 + gridsize * ilev;

      // if ( Options::cdoVerbose ) cdoPrint("level %d: idx1=%d idx2=%d wgt1=%g wgt2=%g", ilev, idx1, idx2, wgt1, wgt2);

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
CPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, int *lev_idx1, int *lev_idx2,
                double *lev_wgt1, double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;

      // if ( Options::cdoVerbose ) cdoPrint("level %d: idx1=%d idx2=%d wgt1=%g wgt2=%g", ilev, idx1, idx2, wgt1, wgt2);

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
GPUs_vert_interp_lev(const size_t gridsize, const float missval,
	       	float *vardata1, float *vardata2,
	       	const int nlev2, int *lev_idx1, int *lev_idx2,
                float *lev_wgt1, float *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      float *var2 = vardata2 + gridsize * ilev;

      // if ( Options::cdoVerbose ) cdoPrint("level %d: idx1=%d idx2=%d wgt1=%g wgt2=%g", ilev, idx1, idx2, wgt1, wgt2);

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
OLD_GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, int *lev_idx1, int *lev_idx2,
                double *lev_wgt1, double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;

      // if ( Options::cdoVerbose ) cdoPrint("level %d: idx1=%d idx2=%d wgt1=%g wgt2=%g", ilev, idx1, idx2, wgt1, wgt2);

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
GPUd_vert_interp_lev(const size_t gridsize, const double missval,
	       	double *vardata1, double *vardata2,
	       	const int nlev2, int *lev_idx1, int *lev_idx2,
                double *lev_wgt1, double *lev_wgt2)
{
  for (int ilev = 0; ilev < nlev2; ++ilev)
    {
      const auto idx1 = lev_idx1[ilev];
      const auto idx2 = lev_idx2[ilev];
      auto wgt1 = lev_wgt1[ilev];
      auto wgt2 = lev_wgt2[ilev];
      double *var2 = vardata2 + gridsize * ilev;

      // if ( Options::cdoVerbose ) cdoPrint("level %d: idx1=%d idx2=%d wgt1=%g wgt2=%g", ilev, idx1, idx2, wgt1, wgt2);

      const double *var1L1 = vardata1 + gridsize * idx1;
      const double *var1L2 = vardata1 + gridsize * idx2;

#ifdef _OPENMP
#pragma omp parallel for default(none) shared(gridsize, var2, var1L1, var1L2, missval, wgt1, wgt2)
#endif
#pragma acc parallel loop
      for (size_t i = 0; i < gridsize; ++i)
        {
//        var2[i] = d_vert_interp_lev_kernel(wgt1, wgt2, var1L1[i], var1L2[i], missval);
          if ((var1L1[i]!=missval)) wgt1 = 0;
          if ((var1L2[i]!=missval)) wgt2 = 0;

          if ((wgt1!=0) && (wgt2!=0))
            {
              var2[i] = missval;
            }
          else if ((wgt1!=0))
            {
              var2[i] = (wgt2 >= 0.5) ? var1L2[i] : missval;
            }
          else if ((wgt2!=0))
            {
              var2[i] = (wgt1 >= 0.5) ? var1L1[i] : missval;
            }
          else
            {
              var2[i] = var1L1[i] * wgt1 + var1L2[i] * wgt2;
            }
        }
    }
}

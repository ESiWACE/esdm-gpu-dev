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
CPUvarrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals)
{
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

  nvals = 0;
  for (size_t i = 0; i < len; ++i)
    {
// printf ("%i %f %f\n",(int) i,array[i],missval);
      if (!DBL_IS_EQUAL(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nvals++;
// printf ("%i %f %f %i\n",(int) i,vmin,vmax,(int) nvals);
        }
    }

  return std::pair<double, double>(vmin, vmax);
}

std::pair<double, double>
GPUvarrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals)
{
  int nv;
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

  nv = 0;
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin) reduction(+:nv)
  for (size_t i = 0; i < len; ++i)
    {
// printf ("%i %f %f\n",(int) i,array[i],missval);
      if (!DBL_IS_EQUAL(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nv++;
// printf ("%i %f %f %i\n",(int) i,vmin,vmax,(int) nvals);
        }
    }
  nvals = nv;

  return std::pair<double, double>(vmin, vmax);
}

#include <cfloat>          // for DBL_MAX, FLT_MAX
#include <utility>         // for std::pair
#include "compare.h"

std::pair<float, float>
CPUs_varrayMinMaxMV(const size_t len, const float *array, const float missval, size_t &nvals)
{
  float vmin = FLT_MAX;
  float vmax = -FLT_MAX;

  nvals = 0;
  for (size_t i = 0; i < len; ++i)
    {
// printf ("%i %f %f\n",(int) i,array[i],missval);
      if (!IS_EQUAL(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nvals++;
// printf ("%i %f %f %i\n",(int) i,vmin,vmax,(int) nvals);
        }
    }

  return std::pair<float, float>(vmin, vmax);
}

std::pair<float, float>
GPUs_varrayMinMaxMV(const size_t len, const float *array, const float missval, size_t &nvals)
{
  int nv;
  float vmin = FLT_MAX;
  float vmax = -FLT_MAX;

  nv = 0;
// MA this causes an internal compiler error in gcc 7.5.0
// the copies seem to be generated automatically anyway by the PGI compiler
//#pragma acc data copy(vmin,vmax) copyin(array[len])
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin) reduction(+:nv)
  for (size_t i = 0; i < len; ++i)
    {
      if (!IS_EQUAL(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nv++;
        }
    }
  nvals = nv;

  return std::pair<float, float>(vmin, vmax);
}

std::pair<double, double>
CPUd_varrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals)
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
GPUd_varrayMinMaxMV(const size_t len, const double *array, const double missval, size_t &nvals)
{
  int nv;
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

  nv = 0;
// MA this causes an internal compiler error in gcc 7.5.0
// the copies seem to be generated automatically anyway by the PGI compiler
//#pragma acc data copy(vmin,vmax) copyin(array[len])
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin) reduction(+:nv)
  for (size_t i = 0; i < len; ++i)
    {
      if (!DBL_IS_EQUAL(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nv++;
        }
    }
  nvals = nv;

  return std::pair<double, double>(vmin, vmax);
}

std::pair<float, float>
CPUs_varrayMinMaxMV_noisnan(const size_t len, const float *array, const float missval, size_t &nvals)
{
  float vmin = FLT_MAX;
  float vmax = -FLT_MAX;

  nvals = 0;
  for (size_t i = 0; i < len; ++i)
    {
// printf ("%i %f %f\n",(int) i,array[i],missval);
      if (!DBL_IS_EQUAL_NOISNAN(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nvals++;
// printf ("%i %f %f %i\n",(int) i,vmin,vmax,(int) nvals);
        }
    }

  return std::pair<float, float>(vmin, vmax);
}

std::pair<float, float>
GPUs_varrayMinMaxMV_noisnan(const size_t len, const float *array, const float missval, size_t &nvals)
{
  int nv;
  float vmin = FLT_MAX;
  float vmax = -FLT_MAX;

  nv = 0;
// MA this causes an internal compiler error in gcc 7.5.0
// the copies seem to be generated automatically anyway by the PGI compiler
//#pragma acc data copy(vmin,vmax) copyin(array[len])
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin) reduction(+:nv)
  for (size_t i = 0; i < len; ++i)
    {
      if (!DBL_IS_EQUAL_NOISNAN(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nv++;
        }
    }
  nvals = nv;

  return std::pair<float, float>(vmin, vmax);
}

std::pair<double, double>
CPUd_varrayMinMaxMV_noisnan(const size_t len, const double *array, const double missval, size_t &nvals)
{
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

  nvals = 0;
  for (size_t i = 0; i < len; ++i)
    {
// printf ("%i %f %f\n",(int) i,array[i],missval);
      if (!DBL_IS_EQUAL_NOISNAN(array[i], missval))
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
GPUd_varrayMinMaxMV_noisnan(const size_t len, const double *array, const double missval, size_t &nvals)
{
  int nv;
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

  nv = 0;
// MA this causes an internal compiler error in gcc 7.5.0
// the copies seem to be generated automatically anyway by the PGI compiler
//#pragma acc data copy(vmin,vmax) copyin(array[len])
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin) reduction(+:nv)
  for (size_t i = 0; i < len; ++i)
    {
      if (!DBL_IS_EQUAL_NOISNAN(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nv++;
        }
    }
  nvals = nv;

  return std::pair<double, double>(vmin, vmax);
}

std::pair<float, float>
CPUs_varrayMinMaxMV_nomissval(const size_t len, const float *array, const float missval, size_t &nvals)
{
  float vmin = FLT_MAX;
  float vmax = -FLT_MAX;

  nvals = len;
  for (size_t i = 0; i < len; ++i)
    {
      if (array[i] < vmin) vmin = array[i];
      if (array[i] > vmax) vmax = array[i];
    }

  return std::pair<float, float>(vmin, vmax);
}

std::pair<float, float>
GPUs_varrayMinMaxMV_nomissval(const size_t len, const float *array, const float missval, size_t &nvals)
{
  float vmin = FLT_MAX;
  float vmax = -FLT_MAX;

// MA this causes an internal compiler error in gcc 7.5.0
// the copies seem to be generated automatically anyway by the PGI compiler
//#pragma acc data copy(vmin,vmax) copyin(array[len])
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin)
  for (size_t i = 0; i < len; ++i)
    {
      if (array[i] < vmin) vmin = array[i];
      if (array[i] > vmax) vmax = array[i];
    }
  nvals = len;

  return std::pair<float, float>(vmin, vmax);
}

std::pair<double, double>
CPUd_varrayMinMaxMV_nomissval(const size_t len, const double *array, const double missval, size_t &nvals)
{
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

  nvals = len;
  for (size_t i = 0; i < len; ++i)
    {
      if (array[i] < vmin) vmin = array[i];
      if (array[i] > vmax) vmax = array[i];
    }

  return std::pair<double, double>(vmin, vmax);
}

std::pair<double, double>
GPUd_varrayMinMaxMV_nomissval(const size_t len, const double *array, const double missval, size_t &nvals)
{
  double vmin = DBL_MAX;
  double vmax = -DBL_MAX;

// MA this causes an internal compiler error in gcc 7.5.0
// the copies seem to be generated automatically anyway by the PGI compiler
//#pragma acc data copy(vmin,vmax) copyin(array[len])
#pragma acc parallel loop reduction(max:vmax) reduction(min:vmin)
  for (size_t i = 0; i < len; ++i)
    {
      if (array[i] < vmin) vmin = array[i];
      if (array[i] > vmax) vmax = array[i];
    }
  nvals = len;

  return std::pair<double, double>(vmin, vmax);
}

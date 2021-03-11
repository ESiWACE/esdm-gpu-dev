/*
  This file is part of CDO. CDO is a collection of Operators to
  manipulate and analyse Climate model Data.

  Copyright (C) 2003-2020 Uwe Schulzweida, <uwe.schulzweida AT mpimet.mpg.de>
  See COPYING file for copying and redistribution conditions.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; version 2 of the License.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"  // __restrict__
#endif

#include <cfloat>
#include <cfenv>
#include <cassert>

#include "dev_compare.h"
#include "dev_array.h"

//#pragma STDC FENV_ACCESS ON

const char *
fpe_errstr(int fpeRaised)
{
  const char *errstr = nullptr;

  // clang-format off
  if      (fpeRaised & FE_DIVBYZERO) errstr = "division by zero";
  else if (fpeRaised & FE_INEXACT)   errstr = "inexact result";
  else if (fpeRaised & FE_INVALID)   errstr = "invalid result";
  else if (fpeRaised & FE_OVERFLOW)  errstr = "overflow";
  else if (fpeRaised & FE_UNDERFLOW) errstr = "underflow";
  // clang-format on

  return errstr;
}

template <typename T>
MinMax
varrayMinMaxMV(const size_t len, const T *array, const T missval)
{
  auto vmin = DBL_MAX;
  auto vmax = -DBL_MAX;

  size_t nvals = 0;
  for (size_t i = 0; i < len; ++i)
    {
      if (!DBL_IS_EQUAL(array[i], missval))
        {
          if (array[i] < vmin) vmin = array[i];
          if (array[i] > vmax) vmax = array[i];
          nvals++;
        }
    }

  return MinMax(vmin, vmax, nvals);
}

// Explicit instantiation
template MinMax varrayMinMaxMV(const size_t len, const float *array, const float missval);
template MinMax varrayMinMaxMV(const size_t len, const double *array, const double missval);

template <typename T>
MinMax
varrayMinMaxMV(const size_t len, const Varray<T> &v, const T missval)
{
  return varrayMinMaxMV(len, v.data(), missval);
}

// Explicit instantiation
template MinMax varrayMinMaxMV(const size_t len, const Varray<float> &v, const float missval);
template MinMax varrayMinMaxMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
MinMaxSum
varrayMinMaxSum(const size_t len, const Varray<T> &v, MinMaxSum mms)
{
  auto rmin = mms.min;
  auto rmax = mms.max;
  auto rsum = mms.sum;

#ifndef __ICC // wrong result with icc19
#ifdef HAVE_OPENMP4
#pragma omp simd reduction(min:rmin) reduction(max:rmax) reduction(+:rsum)
#endif
#endif
  for (size_t i = 0; i < len; ++i)
    {
      const double val = v[i];
      if (val < rmin) rmin = val;
      if (val > rmax) rmax = val;
      rsum += val;
    }

  return MinMaxSum(rmin, rmax, rsum, len);
}

// Explicit instantiation
template MinMaxSum varrayMinMaxSum(const size_t len, const Varray<float> &v, MinMaxSum mms);
template MinMaxSum varrayMinMaxSum(const size_t len, const Varray<double> &v, MinMaxSum mms);

template <typename T>
MinMaxSum
varrayMinMaxSumMV(const size_t len, const Varray<T> &v, const T missval, MinMaxSum mms)
{
  auto rmin = mms.min;
  auto rmax = mms.max;
  auto rsum = mms.sum;

  size_t nvals = 0;
  if (std::isnan(missval))
    {
      for (size_t i = 0; i < len; ++i)
        {
          const double val = v[i];
          if (!DBL_IS_EQUAL(val, missval))
            {
              if (val < rmin) rmin = val;
              if (val > rmax) rmax = val;
              rsum += val;
              nvals++;
            }
        }
    }
  else
    {
#ifndef __ICC // wrong result with icc19
#ifdef HAVE_OPENMP4
#pragma omp simd reduction(min:rmin) reduction(max:rmax) reduction(+:rsum) reduction(+:nvals)
#endif
#endif
      for (size_t i = 0; i < len; ++i)
        {
          const double val = v[i];
          if (IS_NOT_EQUAL(val, missval))
            {
              if (val < rmin) rmin = val;
              if (val > rmax) rmax = val;
              rsum += val;
              nvals++;
            }
        }
    }

  if (nvals == 0 && IS_EQUAL(rmin, DBL_MAX)) rmin = missval;
  if (nvals == 0 && IS_EQUAL(rmax, -DBL_MAX)) rmax = missval;

  return MinMaxSum(rmin, rmax, rsum, nvals);
}

// Explicit instantiation
template MinMaxSum varrayMinMaxSumMV(const size_t len, const Varray<float> &v, const float missval, MinMaxSum mms);
template MinMaxSum varrayMinMaxSumMV(const size_t len, const Varray<double> &v, const double missval, MinMaxSum mms);

MinMaxMean
varrayMinMaxMean(const size_t len, const Varray<double> &v)
{
  auto mms = varrayMinMaxSum(len, v, MinMaxSum());
  const auto rmean = len != 0 ? mms.sum / static_cast<double>(len) : 0.0;
  return MinMaxMean(mms.min, mms.max, rmean, len);
}

MinMaxMean
varrayMinMaxMeanMV(const size_t len, const Varray<double> &v, const double missval)
{
  auto mms = varrayMinMaxSumMV(len, v, missval, MinMaxSum());
  const auto rmean = mms.n != 0 ? mms.sum / static_cast<double>(mms.n) : missval;
  return MinMaxMean(mms.min, mms.max, rmean, mms.n);
}

MinMax
arrayMinMaxMask(const size_t len, const double *const array, const Varray<int> &mask)
{
  auto zmin = DBL_MAX;
  auto zmax = -DBL_MAX;

  if (!mask.empty())
    {
      for (size_t i = 0; i < len; ++i)
        {
          if (mask[i] == 0)
            {
              if (array[i] < zmin) zmin = array[i];
              if (array[i] > zmax) zmax = array[i];
            }
        }
    }
  else
    {
      for (size_t i = 0; i < len; ++i)
        {
          if (array[i] < zmin) zmin = array[i];
          if (array[i] > zmax) zmax = array[i];
        }
    }

  return MinMax(zmin, zmax);
}

void
arrayAddArray(const size_t len, double *__restrict__ array1, const double *__restrict__ array2)
{
  //#ifdef  _OPENMP
  //#pragma omp parallel for default(none) shared(array1,array2)
  //#endif
  for (size_t i = 0; i < len; ++i) array1[i] += array2[i];
}

void
arrayAddArrayMV(const size_t len, double *__restrict__ array1, const double *__restrict__ array2, const double missval)
{
  if (std::isnan(missval))
    {
      for (size_t i = 0; i < len; i++)
        if (!DBL_IS_EQUAL(array2[i], missval))
          {
            if (!DBL_IS_EQUAL(array1[i], missval))
              array1[i] += array2[i];
            else
              array1[i] = array2[i];
          }
    }
  else
    {
      for (size_t i = 0; i < len; i++)
        if (IS_NOT_EQUAL(array2[i], missval))
          {
            if (IS_NOT_EQUAL(array1[i], missval))
              array1[i] += array2[i];
            else
              array1[i] = array2[i];
          }
    }
}

template <typename T>
size_t
arrayNumMV(const size_t len, const T *__restrict__ array, const T missval)
{
  size_t nmiss = 0;

  if (std::isnan(missval))
    {
      for (size_t i = 0; i < len; ++i)
        if (DBL_IS_EQUAL(array[i], missval)) nmiss++;
    }
  else
    {
      for (size_t i = 0; i < len; ++i)
        if (IS_EQUAL(array[i], missval)) nmiss++;
    }

  return nmiss;
}

// Explicit instantiation
template size_t arrayNumMV(const size_t len, const float *array, const float missval);
template size_t arrayNumMV(const size_t len, const double *array, const double missval);

template <typename T>
size_t
varrayNumMV(const size_t len, const Varray<T> &v, const T missval)
{
  size_t nmiss = 0;

  if (std::isnan(missval))
    {
      for (size_t i = 0; i < len; ++i)
        if (DBL_IS_EQUAL(v[i], missval)) nmiss++;
    }
  else
    {
      for (size_t i = 0; i < len; ++i)
        if (IS_EQUAL(v[i], missval)) nmiss++;
    }

  return nmiss;
}

// Explicit instantiation
template size_t varrayNumMV(const size_t len, const Varray<float> &v, const float missval);
template size_t varrayNumMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
MinMax
varrayMinMax(const size_t len, const T *__restrict__ array)
{
  auto vmin = DBL_MAX;
  auto vmax = -DBL_MAX;

#ifndef __ICC // wrong result with icc19
#ifdef HAVE_OPENMP4
#pragma omp simd reduction(min:vmin) reduction(max:vmax)
#endif
#endif
  for (size_t i = 0; i < len; ++i)
    {
      if (array[i] < vmin) vmin = array[i];
      if (array[i] > vmax) vmax = array[i];
    }

  return MinMax(vmin, vmax);
}

// Explicit instantiation
template MinMax varrayMinMax(const size_t len, const float *array);
template MinMax varrayMinMax(const size_t len, const double *array);

template <typename T>
MinMax
varrayMinMax(const size_t len, const Varray<T> &v)
{
  return varrayMinMax(len, v.data());
}

// Explicit instantiation
template MinMax varrayMinMax(const size_t len, const Varray<float> &v);
template MinMax varrayMinMax(const size_t len, const Varray<double> &v);

template <typename T>
MinMax
varrayMinMax(const Varray<T> &v)
{
  auto vmin = DBL_MAX;
  auto vmax = -DBL_MAX;

  const auto len = v.size();
#ifndef __ICC // wrong result with icc19
#ifdef HAVE_OPENMP4
#pragma omp simd reduction(min:vmin) reduction(max:vmax)
#endif
#endif
  for (size_t i = 0; i < len; ++i)
    {
      if (v[i] < vmin) vmin = v[i];
      if (v[i] > vmax) vmax = v[i];
    }

  return MinMax(vmin, vmax);
}

// Explicit instantiation
template MinMax varrayMinMax(const Varray<float> &v);
template MinMax varrayMinMax(const Varray<double> &v);

template <typename T>
double
varrayMin(const size_t len, const Varray<T> &v)
{
  assert(v.size() >= len);

  auto min = v[0];

  for (size_t i = 0; i < len; ++i)
    if (v[i] < min) min = v[i];

  return min;
}

// Explicit instantiation
template double varrayMin(const size_t len, const Varray<float> &v);
template double varrayMin(const size_t len, const Varray<double> &v);

template <typename T>
double
varrayMax(const size_t len, const Varray<T> &v)
{
  assert(v.size() >= len);

  auto max = v[0];

  for (size_t i = 0; i < len; ++i)
    if (v[i] > max) max = v[i];

  return max;
}

// Explicit instantiation
template double varrayMax(const size_t len, const Varray<float> &v);
template double varrayMax(const size_t len, const Varray<double> &v);

template <typename T>
double
varrayRange(const size_t len, const Varray<T> &v)
{
  assert(v.size() >= len);

  auto min = v[0];
  auto max = v[0];

  for (size_t i = 0; i < len; ++i)
    {
      if (v[i] < min) min = v[i];
      if (v[i] > max) max = v[i];
    }

  return (max - min);
}

// Explicit instantiation
template double varrayRange(const size_t len, const Varray<float> &v);
template double varrayRange(const size_t len, const Varray<double> &v);

template <typename T>
double
varrayMinMV(const size_t len, const Varray<T> &v, const T missval)
{
  assert(v.size() >= len);

  double min = DBL_MAX;

  for (size_t i = 0; i < len; ++i)
    if (!DBL_IS_EQUAL(v[i], missval))
      if (v[i] < min) min = v[i];

  if (IS_EQUAL(min, DBL_MAX)) min = missval;

  return min;
}

// Explicit instantiation
template double varrayMinMV(const size_t len, const Varray<float> &v, const float missval);
template double varrayMinMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
double
varrayMaxMV(const size_t len, const Varray<T> &v, const T missval)
{
  assert(v.size() >= len);

  double max = -DBL_MAX;

  for (size_t i = 0; i < len; ++i)
    if (!DBL_IS_EQUAL(v[i], missval))
      if (v[i] > max) max = v[i];

  if (IS_EQUAL(max, -DBL_MAX)) max = missval;

  return max;
}

// Explicit instantiation
template double varrayMaxMV(const size_t len, const Varray<float> &v, const float missval);
template double varrayMaxMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
double
varrayRangeMV(const size_t len, const Varray<T> &v, const T missval)
{
  assert(v.size() >= len);

  double min = DBL_MAX;
  double max = -DBL_MAX;

  for (size_t i = 0; i < len; ++i)
    {
      if (!DBL_IS_EQUAL(v[i], missval))
        {
          if (v[i] < min) min = v[i];
          if (v[i] > max) max = v[i];
        }
    }

  return (IS_EQUAL(min, DBL_MAX) && IS_EQUAL(max, -DBL_MAX)) ? missval : max - min;
}

// Explicit instantiation
template double varrayRangeMV(const size_t len, const Varray<float> &v, const float missval);
template double varrayRangeMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
double
varraySum(const size_t len, const Varray<T> &v)
{
  assert(v.size() >= len);

  double sum = 0.0;
  for (size_t i = 0; i < len; ++i) sum += v[i];

  return sum;
}

// Explicit instantiation
template double varraySum(const size_t len, const Varray<float> &v);
template double varraySum(const size_t len, const Varray<double> &v);

template <typename T>
double
varraySumMV(const size_t len, const Varray<T> &v, const T missval)
{
  assert(v.size() >= len);

  double sum = 0.0;
  size_t nvals = 0;

  if (std::isnan(missval))
    {
      for (size_t i = 0; i < len; ++i)
        if (!DBL_IS_EQUAL(v[i], missval))
          {
            sum += v[i];
            nvals++;
          }
    }
  else
    {
      for (size_t i = 0; i < len; ++i)
        if (IS_NOT_EQUAL(v[i], missval))
          {
            sum += v[i];
            nvals++;
          }
    }

  if (!nvals) sum = missval;

  return sum;
}

// Explicit instantiation
template double varraySumMV(const size_t len, const Varray<float> &v, const float missval);
template double varraySumMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
double
varrayMean(const size_t len, const Varray<T> &v)
{
  assert(v.size() >= len);

  const auto sum = varraySum(len, v);

  return sum / len;
}

// Explicit instantiation
template double varrayMean(const size_t len, const Varray<float> &v);
template double varrayMean(const size_t len, const Varray<double> &v);

template <typename T>
double
varrayMeanMV(const size_t len, const Varray<T> &v, const T missval)
{
  assert(v.size() >= len);

  double sum = 0.0, sumw = 0.0;

  for (size_t i = 0; i < len; ++i)
    if (!DBL_IS_EQUAL(v[i], missval))
      {
        sum += v[i];
        sumw += 1;
      }

  double missval1 = missval, missval2 = missval;
  return DIVMN(sum, sumw);
}

// Explicit instantiation
template double varrayMeanMV(const size_t len, const Varray<float> &v, const float missval);
template double varrayMeanMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
double
varrayWeightedMean(const size_t len, const Varray<T> &v, const Varray<double> &w, const T missval)
{
  assert(v.size() >= len);
  assert(w.size() >= len);

  double sum = 0.0, sumw = 0.0;

  for (size_t i = 0; i < len; ++i)
    {
      sum += w[i] * v[i];
      sumw += w[i];
    }

  return IS_EQUAL(sumw, 0.) ? missval : sum / sumw;
}

// Explicit instantiation
template double varrayWeightedMean(const size_t len, const Varray<float> &v, const Varray<double> &w, const float missval);
template double varrayWeightedMean(const size_t len, const Varray<double> &v, const Varray<double> &w, const double missval);

template <typename T>
double
varrayWeightedMeanMV(const size_t len, const Varray<T> &v, const Varray<double> &w, const T missval)
{
  assert(v.size() >= len);
  assert(w.size() >= len);

  const double missval1 = missval, missval2 = missval;
  double sum = 0.0, sumw = 0.0;

  for (size_t i = 0; i < len; ++i)
    if (!DBL_IS_EQUAL(v[i], missval1) && !DBL_IS_EQUAL(w[i], missval1))
      {
        sum += w[i] * v[i];
        sumw += w[i];
      }

  return DIVMN(sum, sumw);
}

// Explicit instantiation
template double varrayWeightedMeanMV(const size_t len, const Varray<float> &v, const Varray<double> &w, const float missval);
template double varrayWeightedMeanMV(const size_t len, const Varray<double> &v, const Varray<double> &w, const double missval);

template <typename T>
double
varrayAvgMV(const size_t len, const Varray<T> &v, const T missval)
{
  assert(v.size() >= len);

  const double missval1 = missval, missval2 = missval;
  double sum = 0.0, sumw = 0.0;

  for (size_t i = 0; i < len; ++i)
    {
      sum = ADDMN(sum, v[i]);
      sumw += 1;
    }

  return DIVMN(sum, sumw);
}

// Explicit instantiation
template double varrayAvgMV(const size_t len, const Varray<float> &v, const float missval);
template double varrayAvgMV(const size_t len, const Varray<double> &v, const double missval);

template <typename T>
double
varrayWeightedAvgMV(const size_t len, const Varray<T> &v, const Varray<double> &w, const T missval)
{
  assert(v.size() >= len);
  assert(w.size() >= len);

  const double missval1 = missval, missval2 = missval;
  double sum = 0.0, sumw = 0.0;

  for (size_t i = 0; i < len; ++i)
    if (!DBL_IS_EQUAL(w[i], missval))
      {
        sum = ADDMN(sum, MULMN(w[i], v[i]));
        sumw = ADDMN(sumw, w[i]);
      }

  return DIVMN(sum, sumw);
}

// Explicit instantiation
template double varrayWeightedAvgMV(const size_t len, const Varray<float> &v, const Varray<double> &w, const float missval);
template double varrayWeightedAvgMV(const size_t len, const Varray<double> &v, const Varray<double> &w, const double missval);

template <typename T>
static void
varrayPrevarsum0(size_t len, const Varray<T> &v, double &rsum, double &rsumw)
{
  rsum = 0.0;
  for (size_t i = 0; i < len; i++)
    {
      rsum += v[i];
    }
  rsumw = len;
}

template <typename T>
static void
varrayPrevarsum0MV(size_t len, const Varray<T> &v, double missval, double &rsum, double &rsumw)
{
  rsum = rsumw = 0.0;
  for (size_t i = 0; i < len; i++)
    if (!DBL_IS_EQUAL(v[i], missval))
      {
        rsum += v[i];
        rsumw += 1.0;
      }
}

template <typename T>
static void
varrayPrevarsum(size_t len, const Varray<T> &v,
                double &rsum, double &rsumw, double &rsumq, double &rsumwq)
{
  rsum = rsumq = 0.0;
  for (size_t i = 0; i < len; i++)
    {
      const double vd = v[i];
      rsum += vd;
      rsumq += vd * vd;
    }
  rsumw = len;
  rsumwq = len;
}

template <typename T>
static void
varrayPrevarsumMV(size_t len, const Varray<T> &v, T missval,
                  double &rsum, double &rsumw, double &rsumq, double &rsumwq)
{
  rsum = rsumq = rsumw = rsumwq = 0.0;
  for (size_t i = 0; i < len; i++)
    if (!DBL_IS_EQUAL(v[i], missval))
      {
        const double vd = v[i];
        rsum += vd;
        rsumq += vd * vd;
        rsumw += 1.0;
        rsumwq += 1.0;
      }
}

template <typename T>
double
varrayVar(size_t len, const Varray<T> &v, size_t nmiss, T missval)
{
  double rsum = 0.0, rsumw = 0.0, rsumq = 0.0, rsumwq = 0.0;
  if (nmiss > 0)
    varrayPrevarsumMV(len, v, missval, rsum, rsumw, rsumq, rsumwq);
  else
    varrayPrevarsum(len, v, rsum, rsumw, rsumq, rsumwq);

  auto rvar = IS_NOT_EQUAL(rsumw, 0.0) ? (rsumq * rsumw - rsum * rsum) / (rsumw * rsumw) : missval;
  if (rvar < 0.0 && rvar > -1.e-5) rvar = 0.0;

  return rvar;
}

// Explicit instantiation
template double varrayVar(const size_t len, const Varray<float> &v, size_t nmiss, const float missval);
template double varrayVar(const size_t len, const Varray<double> &v, size_t nmiss, const double missval);

template <typename T>
double
varrayVar1(size_t len, const Varray<T> &v, size_t nmiss, T missval)
{
  double rsum = 0.0, rsumw = 0.0, rsumq = 0.0, rsumwq = 0.0;
  if (nmiss > 0)
    varrayPrevarsumMV(len, v, missval, rsum, rsumw, rsumq, rsumwq);
  else
    varrayPrevarsum(len, v, rsum, rsumw, rsumq, rsumwq);

  auto rvar = (rsumw * rsumw > rsumwq) ? (rsumq * rsumw - rsum * rsum) / (rsumw * rsumw - rsumwq) : missval;
  if (rvar < 0.0 && rvar > -1.e-5) rvar = 0.0;

  return rvar;
}

// Explicit instantiation
template double varrayVar1(const size_t len, const Varray<float> &v, size_t nmiss, const float missval);
template double varrayVar1(const size_t len, const Varray<double> &v, size_t nmiss, const double missval);

template <typename T>
static void
varrayWeightedPrevarsum(size_t len, const Varray<T> &v, const Varray<double> &w,
                        double &rsum, double &rsumw, double &rsumq, double &rsumwq)
{
  rsum = rsumq = rsumw = rsumwq = 0.0;
  for (size_t i = 0; i < len; i++)
    {
      const double vd = v[i];
      rsum += w[i] * vd;
      rsumq += w[i] * vd * vd;
      rsumw += w[i];
      rsumwq += w[i] * w[i];
    }
}

template <typename T>
static void
varrayWeightedPrevarsumMV(size_t len, const Varray<T> &v, const Varray<double> &w, double missval,
                          double &rsum, double &rsumw, double &rsumq, double &rsumwq)
{
  rsum = rsumq = rsumw = rsumwq = 0.0;
  for (size_t i = 0; i < len; i++)
    if (!DBL_IS_EQUAL(v[i], missval) && !DBL_IS_EQUAL(w[i], missval))
      {
        const double vd = v[i];
        rsum += w[i] * vd;
        rsumq += w[i] * vd * vd;
        rsumw += w[i];
        rsumwq += w[i] * w[i];
      }
}

template <typename T>
double
varrayWeightedVar(size_t len, const Varray<T> &v, const Varray<double> &w, size_t nmiss, T missval)
{
  double rsum = 0.0, rsumw = 0.0, rsumq = 0.0, rsumwq = 0.0;
  if (nmiss > 0)
    varrayWeightedPrevarsumMV(len, v, w, missval, rsum, rsumw, rsumq, rsumwq);
  else
    varrayWeightedPrevarsum(len, v, w, rsum, rsumw, rsumq, rsumwq);

  auto rvar = IS_NOT_EQUAL(rsumw, 0) ? (rsumq * rsumw - rsum * rsum) / (rsumw * rsumw) : missval;
  if (rvar < 0.0 && rvar > -1.e-5) rvar = 0.0;

  return rvar;
}

// Explicit instantiation
template double varrayWeightedVar(const size_t len, const Varray<float> &v, const Varray<double> &w, size_t nmiss, const float missval);
template double varrayWeightedVar(const size_t len, const Varray<double> &v, const Varray<double> &w, size_t nmiss, const double missval);

template <typename T>
double
varrayWeightedVar1(size_t len, const Varray<T> &v, const Varray<double> &w, size_t nmiss, T missval)
{
  double rsum = 0.0, rsumw = 0.0, rsumq = 0.0, rsumwq = 0.0;
  if (nmiss > 0)
    varrayWeightedPrevarsumMV(len, v, w, missval, rsum, rsumw, rsumq, rsumwq);
  else
    varrayWeightedPrevarsum(len, v, w, rsum, rsumw, rsumq, rsumwq);

  auto rvar = (rsumw * rsumw > rsumwq) ? (rsumq * rsumw - rsum * rsum) / (rsumw * rsumw - rsumwq) : missval;
  if (rvar < 0.0 && rvar > -1.e-5) rvar = 0.0;

  return rvar;
}

// Explicit instantiation
template double varrayWeightedVar1(const size_t len, const Varray<float> &v, const Varray<double> &w, size_t nmiss, const float missval);
template double varrayWeightedVar1(const size_t len, const Varray<double> &v, const Varray<double> &w, size_t nmiss, const double missval);

template <typename T>
static void
varrayPrekurtsum(size_t len, const Varray<T> &v, const double mean,
                 double &rsum3w, double &rsum4w, double &rsum2diff, double &rsum4diff)
{
  rsum2diff = rsum4diff = 0.0;
  for (size_t i = 0; i < len; i++)
    {
      const double vdiff = v[i] - mean;
      rsum2diff += vdiff * vdiff;
      rsum4diff += vdiff * vdiff * vdiff * vdiff;
    }
  rsum3w = len;
  rsum4w = len;
}

template <typename T>
static void
varrayPrekurtsumMV(size_t len, const Varray<T> &v, T missval, const double mean,
                   double &rsum3w, double &rsum4w, double &rsum2diff, double &rsum4diff)
{
  rsum3w = rsum4w = rsum2diff = rsum4diff = 0.0;
  for (size_t i = 0; i < len; i++)
    if (!DBL_IS_EQUAL(v[i], missval))
      {
        const double vdiff = v[i] - mean;
        rsum2diff += vdiff * vdiff;
        rsum4diff += vdiff * vdiff * vdiff * vdiff;
        rsum3w += 1;
        rsum4w += 1;
      }
}

template <typename T>
double
varrayKurt(size_t len, const Varray<T> &v, size_t nmiss, T missval)
{
  double rsum3w;  // 3rd moment variables
  double rsum4w;  // 4th moment variables
  double rsum2diff, rsum4diff;
  double rsum, rsumw;
  if (nmiss > 0)
    {
      varrayPrevarsum0MV(len, v, missval, rsum, rsumw);
      varrayPrekurtsumMV(len, v, missval, (rsum / rsumw), rsum3w, rsum4w, rsum2diff, rsum4diff);
    }
  else
    {
      varrayPrevarsum0(len, v, rsum, rsumw);
      varrayPrekurtsum(len, v, (rsum / rsumw), rsum3w, rsum4w, rsum2diff, rsum4diff);
    }

  if (IS_EQUAL(rsum3w, 0.0) || IS_EQUAL(rsum2diff, 0.0)) return missval;

  auto rkurt = ((rsum4diff / rsum3w) / std::pow(rsum2diff / rsum3w, 2)) - 3.0;
  if (rkurt < 0.0 && rkurt > -1.e-5) rkurt = 0.0;

  return rkurt;
}

// Explicit instantiation
template double varrayKurt(const size_t len, const Varray<float> &v, size_t nmiss, const float missval);
template double varrayKurt(const size_t len, const Varray<double> &v, size_t nmiss, const double missval);

template <typename T>
static void
varrayPreskewsum(size_t len, const Varray<T> &v, const double mean,
                 double &rsum3w, double &rsum4w, double &rsum3diff, double &rsum2diff)
{
  rsum3diff = rsum2diff = 0.0;
  for (size_t i = 0; i < len; i++)
    {
      const double vdiff = v[i] - mean;
      rsum3diff += vdiff * vdiff * vdiff;
      rsum2diff += vdiff * vdiff;
    }
  rsum3w = len;
  rsum4w = len;
}

template <typename T>
static void
varrayPreskewsumMV(size_t len, const Varray<T> &v, T missval, const double mean,
                   double &rsum3w, double &rsum4w, double &rsum3diff, double &rsum2diff)
{
  rsum3w = rsum4w = rsum3diff = rsum2diff = 0.0;
  for (size_t i = 0; i < len; i++)
    if (!DBL_IS_EQUAL(v[i], missval))
      {
        const double vdiff = v[i] - mean;
        rsum3diff += vdiff * vdiff * vdiff;
        rsum2diff += vdiff * vdiff;
        rsum3w += 1;
        rsum4w += 1;
      }
}

template <typename T>
double
varraySkew(size_t len, const Varray<T> &v, size_t nmiss, T missval)
{
  double rsum3w;  // 3rd moment variables
  double rsum4w;  // 4th moment variables
  double rsum3diff, rsum2diff;
  double rsum, rsumw;
  if (nmiss > 0)
    {
      varrayPrevarsum0MV(len, v, missval, rsum, rsumw);
      varrayPreskewsumMV(len, v, missval, (rsum / rsumw), rsum3w, rsum4w, rsum3diff, rsum2diff);
    }
  else
    {
      varrayPrevarsum0(len, v, rsum, rsumw);
      varrayPreskewsum(len, v, (rsum / rsumw), rsum3w, rsum4w, rsum3diff, rsum2diff);
    }

  if (IS_EQUAL(rsum3w, 0.0) || IS_EQUAL(rsum3w, 1.0) || IS_EQUAL(rsum2diff, 0.0)) return missval;

  auto rskew = (rsum3diff / rsum3w) / std::pow((rsum2diff) / (rsum3w - 1.0), 1.5);
  if (rskew < 0.0 && rskew > -1.e-5) rskew = 0.0;

  return rskew;
}

// Explicit instantiation
template double varraySkew(const size_t len, const Varray<float> &v, size_t nmiss, const float missval);
template double varraySkew(const size_t len, const Varray<double> &v, size_t nmiss, const double missval);

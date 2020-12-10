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
#ifndef ARRAY_H
#define ARRAY_H

#include <iostream>  // cerr
#include <vector>
#include <cstddef>
#include <cfloat>
#include "dev_compare.h"

// #define CHECK_UNUSED_VECTOR 1

#ifdef CHECK_UNUSED_VECTOR
template <typename T>
class
#ifdef __GNUG__
__attribute__((warn_unused))
#endif
CheckVector
{
 public:
  T *ptr;
  size_t m_count;

  CheckVector() : ptr(dummy), m_count(0) { }
  CheckVector(size_t count) : ptr(dummy), m_count(count) { }
  CheckVector(size_t count, const T& value) : ptr(dummy), m_count(count) { ptr[0] = value; }

  T * begin() noexcept { return &ptr[0]; }
  T * end() noexcept { return &ptr[0] + 1; }
  const T * begin() const noexcept { return &ptr[0]; }
  const T * end() const noexcept { return &ptr[0] + 1; }
  
  bool empty() const { return true; }
  size_t size() const { return m_count; }
  void clear() { }
  void shrink_to_fit() { }

  void resize(size_t count) { m_count = count; }
  void resize(size_t count, const T& value) { m_count = count; ptr[0] = value; }

  void reserve(size_t new_cap) { (void) new_cap; }
  void push_back(const T& value) { ptr[0] = value; }

  T * data() noexcept { return ptr; }
  const T* data() const noexcept { return ptr; }

  T& operator[](size_t pos) { (void) pos; return ptr[0]; }
  const T & operator[](size_t pos) const { (void) pos; return ptr[0]; }
  CheckVector& operator=( const CheckVector& other ) { (void)other; return *this; }
  CheckVector& operator=( CheckVector&& other ) { (void)other;  return *this; }
  // Copy constructor
  CheckVector(const CheckVector &v2) { ptr[0] = v2.ptr[0]; }

  bool operator==(const CheckVector& other) { (void)other; return true; }

 private:
  T dummy[1];
};

template <typename T>
using Varray = CheckVector<T>;

#else

template <typename T>
using Varray = std::vector<T>;

#endif

template <typename T>
using Varray2D = Varray<Varray<T>>;

template <typename T>
using Varray3D = Varray2D<Varray<T>>;

template <typename T>
using Varray4D = Varray3D<Varray<T>>;


struct MinMax
{
  double min = DBL_MAX;
  double max = -DBL_MAX;
  size_t n = 0;
  MinMax() {};
  MinMax(double rmin, double rmax, size_t rn) : min(rmin), max(rmax), n(rn) {};
  MinMax(double rmin, double rmax) : min(rmin), max(rmax), n(0) {};
};

struct MinMaxSum : MinMax
{
  double sum = 0;
  MinMaxSum() {};
  MinMaxSum(double rmin, double rmax, double rsum, size_t rn) : sum(rsum) { min = rmin; max = rmax; n = rn; };
  MinMaxSum(double rmin, double rmax, double rsum) : sum(rsum) { min = rmin; max = rmax; n = 0; };
};

struct MinMaxMean : MinMax
{
  double mean = 0;
  MinMaxMean() {};
  MinMaxMean(double rmin, double rmax, double rmean, size_t rn) : mean(rmean) { min = rmin; max = rmax; n = rn; };
  MinMaxMean(double rmin, double rmax, double rmean) : mean(rmean) { min = rmin; max = rmax; n = 0; };
};

#define IX2D(y, x, nx) ((y) * (nx) + (x))

template <typename T>
inline void
varrayFree(Varray<T> &v)
{
  v.clear();
  v.shrink_to_fit();
}

#define varrayResize(p, s) varray_resize((p), (s), __FILE__, __LINE__)
#define varrayResizeInit(p, s, v) varray_resize((p), (s), (v), __FILE__, __LINE__)

template <typename T>
inline void
varray_resize(Varray<T> &v, size_t count, const char *file, int line)
{
  try
    {
      v.resize(count);
    }
  catch (const std::exception &e)
    {
      std::cerr << "Exception caught when trying to allocate " << count << " vector elements: "
                << e.what() << " in " << file << ":" << line << '\n';
      throw;
    }
}

template <typename T>
inline void
varray_resize(Varray<T> &v, size_t count, T value, const char *file, int line)
{
  try
    {
      v.resize(count, value);
    }
  catch (const std::exception &e)
    {
      std::cerr << "Exception caught when trying to allocate " << count << " vector elements: "
                << e.what() << " in " << file << ":" << line << '\n';
      throw;
    }
}

/*
template <class T, size_t ROW, size_t COL>
using Matrix = std::array<std::array<T, COL>, ROW>;
*/

// clang-format off
#define MADDMN(x, y) (DBL_IS_EQUAL((x), missval1) || DBL_IS_EQUAL((y), missval2) ? missval1 : (x) + (y))
#define MSUBMN(x, y) (DBL_IS_EQUAL((x), missval1) || DBL_IS_EQUAL((y), missval2) ? missval1 : (x) - (y))
#define MMULMN(x, y) (DBL_IS_EQUAL((x), 0.) || DBL_IS_EQUAL((y), 0.) \
                      ? 0  : DBL_IS_EQUAL((x), missval1) || DBL_IS_EQUAL((y), missval2) ? missval1 : (x) * (y))
#define MDIVMN(x, y) (DBL_IS_EQUAL((x), missval1) || DBL_IS_EQUAL((y), missval2) || DBL_IS_EQUAL((y), 0.) ? missval1 : (x) / (y))
#define MPOWMN(x, y) (DBL_IS_EQUAL((x), missval1) || DBL_IS_EQUAL((y), missval2) ? missval1 : std::pow((x), (y)))
#define MSQRTMN(x) (DBL_IS_EQUAL((x), missval1) || (x) < 0 ? missval1 : std::sqrt(x))

#define ADD(x, y) ((x) + (y))
#define SUB(x, y) ((x) - (y))
#define MUL(x, y) ((x) * (y))
#define DIV(x, y) (IS_EQUAL((y), 0.) ? missval1 : (x) / (y))
#define POW(x, y) std::pow((x), (y))
#define SQRT(x) std::sqrt(x)

#define ADDM(x, y) (IS_EQUAL((x), missval1) || IS_EQUAL((y), missval2) ? missval1 : (x) + (y))
#define SUBM(x, y) (IS_EQUAL((x), missval1) || IS_EQUAL((y), missval2) ? missval1 : (x) - (y))
#define MULM(x, y)                            \
  (IS_EQUAL((x), 0.) || IS_EQUAL((y), 0.) ? 0 \
                                          : IS_EQUAL((x), missval1) || IS_EQUAL((y), missval2) ? missval1 : (x) * (y))
#define DIVM(x, y) (IS_EQUAL((x), missval1) || IS_EQUAL((y), missval2) || IS_EQUAL((y), 0.) ? missval1 : (x) / (y))
#define POWM(x, y) (IS_EQUAL((x), missval1) || IS_EQUAL((y), missval2) ? missval1 : std::pow((x), (y)))
#define SQRTM(x) (IS_EQUAL((x), missval1) || (x) < 0 ? missval1 : std::sqrt(x))

#define ADDMN(x, y) FADDMN(x, y, missval1, missval2)
#define SUBMN(x, y) FSUBMN(x, y, missval1, missval2)
#define MULMN(x, y) FMULMN(x, y, missval1, missval2)
#define DIVMN(x, y) FDIVMN(x, y, missval1, missval2)
#define POWMN(x, y) FPOWMN(x, y, missval1, missval2)
#define SQRTMN(x) FSQRTMN(x, missval1)

static inline double FADDMN(const double x, const double y, const double missval1, const double missval2)
{
  return MADDMN(x, y);
}
static inline double FSUBMN(const double x, const double y, const double missval1, const double missval2)
{
  return MSUBMN(x, y);
}
static inline double FMULMN(const double x, const double y, const double missval1, const double missval2)
{
  return MMULMN(x, y);
}
static inline double FDIVMN(const double x, const double y, const double missval1, const double missval2)
{
  return MDIVMN(x, y);
}
static inline double FPOWMN(const double x, const double y, const double missval1, const double missval2)
{
  return MPOWMN(x, y);
}
static inline double FSQRTMN(const double x, const double missval1)
{
  return MSQRTMN(x);
}
// clang-format on

const char *fpe_errstr(int fpeRaised);

template <typename T>
MinMaxSum varrayMinMaxSum(size_t len, const Varray<T> &v, MinMaxSum mms);

template <typename T>
MinMaxSum varrayMinMaxSumMV(size_t len, const Varray<T> &v, T missval, MinMaxSum mms);

MinMaxMean varrayMinMaxMean(size_t len, const Varray<double> &v);
MinMaxMean varrayMinMaxMeanMV(size_t len, const Varray<double> &v, double missval);

MinMax arrayMinMaxMask(size_t len, const double *array, const Varray<int> &mask);

void arrayAddArray(size_t len, double *array1, const double *array2);
void arrayAddArrayMV(size_t len, double *array1, const double *array2, double missval);

template <typename T>
void
varrayFill(const size_t len, T *v, const T value)
{
  for (size_t i = 0; i < len; ++i) v[i] = value;
}

template <typename T>
void
varrayFill(Varray<T> &v, const T value)
{
  varrayFill(v.size(), v.data(), value);
}

template <typename T>
void
varrayFill(const size_t len, Varray<T> &v, const T value)
{
  varrayFill(len, v.data(), value);
}

template <typename T>
void
arrayCopy(const size_t len, const T *array1, T *array2)
{
  for (size_t i = 0; i < len; ++i) array2[i] = array1[i];
}

template <typename T>
void
varrayCopy(const size_t len, const T &array1, T &array2)
{
  for (size_t i = 0; i < len; ++i) array2[i] = array1[i];
}

template <typename T>
size_t arrayNumMV(size_t len, const T *array, T missval);

template <typename T>
size_t varrayNumMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
MinMax varrayMinMax(size_t len, const Varray<T> &v);

template <typename T>
MinMax varrayMinMax(size_t len, const T *array);

template <typename T>
MinMax varrayMinMax(const Varray<T> &v);

template <typename T>
MinMax varrayMinMaxMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
MinMax varrayMinMaxMV(size_t len, const T *array, T missval);

template <typename T>
double varrayMin(size_t len, const Varray<T> &v);

template <typename T>
double varrayMax(size_t len, const Varray<T> &v);

template <typename T>
double varrayRange(size_t len, const Varray<T> &v);

template <typename T>
double varrayMinMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
double varrayMaxMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
double varrayRangeMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
double varraySum(size_t len, const Varray<T> &v);

template <typename T>
double varraySumMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
double varrayMean(size_t len, const Varray<T> &v);

template <typename T>
double varrayMeanMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
double varrayWeightedMean(size_t len, const Varray<T> &v, const Varray<double> &w, T missval);

template <typename T>
double varrayWeightedMeanMV(size_t len, const Varray<T> &v, const Varray<double> &w, T missval);

template <typename T>
double varrayAvgMV(size_t len, const Varray<T> &v, T missval);

template <typename T>
double varrayWeightedAvgMV(size_t len, const Varray<T> &v, const Varray<double> &w, T missval);

template <typename T>
double varrayVar(size_t len, const Varray<T> &v, size_t nmiss, T missval);

template <typename T>
double varrayVar1(size_t len, const Varray<T> &v, size_t nmiss, T missval);

template <typename T>
double varrayWeightedVar(size_t len, const Varray<T> &v, const Varray<double> &w, size_t nmiss, T missval);

template <typename T>
double varrayWeightedVar1(size_t len, const Varray<T> &v, const Varray<double> &w, size_t nmiss, T missval);

template <typename T>
double varrayKurt(size_t len, const Varray<T> &v, size_t nmiss, T missval);

template <typename T>
double varraySkew(size_t len, const Varray<T> &v, size_t nmiss, T missval);

#endif  //  ARRAY_H

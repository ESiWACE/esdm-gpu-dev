/*
  This file is part of CDO. CDO is a collection of Operators to
  manipulate and analyse Climate model Data.

  Copyright (C) 2003-2019 Uwe Schulzweida, <uwe.schulzweida AT mpimet.mpg.de>
  See COPYING file for copying and redistribution conditions.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; version 2 of the License.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/

#ifndef COMPARE_H
#define COMPARE_H

#include <cmath>
#include <string.h>

// MA two definitions with and without testing for NaN
#define DBL_IS_EQUAL_NOISNAN(x,y) (!(x < y || y < x))
#define DBL_IS_EQUAL(x, y) (std::isnan(x) || std::isnan(y) ? (std::isnan(x) && std::isnan(y) ? 1 : 0) : !(x < y || y < x))

#define IS_NOT_EQUAL(x, y) (x < y || y < x)
#define IS_EQUAL(x, y) (!IS_NOT_EQUAL(x, y))

#define cmpstr(s1, s2) (strncmp(s1, s2, strlen(s2)))
#define cmpstrlen(s1, s2, len) (strncmp(s1, s2, len = strlen(s2)))

static inline bool
cstrIsEqual(const char *x, const char *y)
{
  return (*x == *y) && strcmp(x, y) == 0;
}

#endif /* COMPARE_H */
